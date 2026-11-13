/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Mutex},
};

use answer::variable_value::VariableValue;
use resource::constants::traversal::{
    FIXED_BATCH_ROWS_MAX, FUNCTION_CALL_CACHE_CALLS_MAX, FUNCTION_CALL_CACHE_ROWS_PER_CALL_MAX,
    FUNCTION_CALL_CACHE_ROWS_TOTAL_MAX,
};

use crate::{batch::FixedBatch, row::MaybeOwnedRow};

/// The rows returned by calls to non-recursive functions during one read pipeline.
/// A call is recorded the second time it is seen, and cached if its function body returned all its rows.
#[derive(Debug, Default)]
pub(crate) struct FunctionCallCache {
    state: Mutex<CacheState>,
}

#[derive(Debug, Default)]
struct CacheState {
    calls: HashMap<FunctionCallKey, CallEntry>,
    total_rows: usize,
}

#[derive(Debug)]
enum CallEntry {
    Seen,
    Uncached,
    Cached(Arc<CallRows>),
}

// The function body receives the multiplicity of the caller's row, and may return it in its rows
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FunctionCallKey {
    executable_id: u64,
    arguments: Vec<VariableValue<'static>>,
    multiplicity: u64,
}

impl FunctionCallKey {
    pub(crate) fn new(executable_id: u64, arguments: Vec<VariableValue<'static>>, multiplicity: u64) -> Self {
        Self { executable_id, arguments, multiplicity }
    }
}

#[derive(Debug, Default)]
pub(crate) struct CallRows {
    width: u32,
    rows: Vec<MaybeOwnedRow<'static>>,
}

impl CallRows {
    pub(crate) fn batch_starting(&self, start: usize) -> Option<FixedBatch> {
        let rows = self.rows.get(start..)?.iter().take(FIXED_BATCH_ROWS_MAX as usize);
        let mut batch = FixedBatch::new(self.width);
        rows.for_each(|row| batch.append(|mut output_row| output_row.copy_from_row(row.as_reference())));
        (!batch.is_empty()).then_some(batch)
    }
}

pub(crate) enum CallLookup {
    Cached(Arc<CallRows>),
    Record(Box<CallRecording>),
    Execute,
}

#[derive(Debug)]
pub(crate) struct CallRecording {
    key: FunctionCallKey,
    // None once the call has returned too many rows to cache
    rows: Option<CallRows>,
}

impl CallRecording {
    pub(crate) fn record(&mut self, batch: &FixedBatch) {
        let Some(rows) = &mut self.rows else { return };
        if rows.rows.len() + batch.len() as usize > FUNCTION_CALL_CACHE_ROWS_PER_CALL_MAX {
            self.rows = None;
            return;
        }
        rows.width = batch.width();
        rows.rows.extend((0..batch.len()).map(|index| batch.get_row(index).into_owned()));
    }
}

impl FunctionCallCache {
    pub(crate) fn lookup(&self, key: FunctionCallKey) -> CallLookup {
        let mut state = self.state.lock().unwrap();
        let is_full = state.total_rows >= FUNCTION_CALL_CACHE_ROWS_TOTAL_MAX;
        let calls_count = state.calls.len();
        match state.calls.entry(key) {
            Entry::Occupied(entry) => match entry.get() {
                CallEntry::Cached(rows) => CallLookup::Cached(rows.clone()),
                CallEntry::Seen if !is_full => CallLookup::Record(Box::new(CallRecording {
                    key: entry.key().clone(),
                    rows: Some(CallRows::default()),
                })),
                CallEntry::Seen | CallEntry::Uncached => CallLookup::Execute,
            },
            Entry::Vacant(entry) => {
                if calls_count < FUNCTION_CALL_CACHE_CALLS_MAX {
                    entry.insert(CallEntry::Seen);
                }
                CallLookup::Execute
            }
        }
    }

    pub(crate) fn complete(&self, recording: Box<CallRecording>) {
        let CallRecording { key, rows } = *recording;
        let mut state = self.state.lock().unwrap();
        let CacheState { calls, total_rows } = &mut *state;
        // another recording of the same call may have completed first
        let Some(entry @ CallEntry::Seen) = calls.get_mut(&key) else { return };
        match rows {
            Some(rows) if *total_rows + rows.rows.len() <= FUNCTION_CALL_CACHE_ROWS_TOTAL_MAX => {
                *total_rows += rows.rows.len();
                *entry = CallEntry::Cached(Arc::new(rows));
            }
            _ => *entry = CallEntry::Uncached,
        }
    }
}

#[cfg(test)]
mod tests {
    use answer::variable_value::VariableValue;
    use compiler::VariablePosition;
    use encoding::value::value::Value;
    use resource::constants::traversal::{FIXED_BATCH_ROWS_MAX, FUNCTION_CALL_CACHE_ROWS_PER_CALL_MAX};

    use super::{CallLookup, FunctionCallCache, FunctionCallKey};
    use crate::{Provenance, batch::FixedBatch, row::MaybeOwnedRow};

    fn key(argument: i64) -> FunctionCallKey {
        FunctionCallKey::new(1, vec![VariableValue::Value(Value::Integer(argument))], 1)
    }

    fn batch(values: impl IntoIterator<Item = i64>) -> FixedBatch {
        let mut batch = FixedBatch::new(1);
        for value in values {
            let row = MaybeOwnedRow::new_owned(vec![VariableValue::Value(Value::Integer(value))], 1, Provenance(0));
            batch.append(|mut output_row| output_row.copy_from_row(row));
        }
        batch
    }

    fn record(cache: &FunctionCallCache, key: FunctionCallKey, rows: i64, complete: bool) {
        let CallLookup::Record(mut recording) = cache.lookup(key) else { panic!("Expected to record the call") };
        for start in (0..rows).step_by(FIXED_BATCH_ROWS_MAX as usize) {
            recording.record(&batch(start..rows.min(start + FIXED_BATCH_ROWS_MAX as i64)));
        }
        if complete {
            cache.complete(recording);
        }
    }

    fn cached_rows(cache: &FunctionCallCache, key: FunctionCallKey) -> Option<Vec<i64>> {
        let CallLookup::Cached(rows) = cache.lookup(key) else { return None };
        let mut values = Vec::new();
        while let Some(batch) = rows.batch_starting(values.len()) {
            for index in 0..batch.len() {
                values.push(batch.get_row(index).get(VariablePosition::new(0)).as_value().clone().unwrap_integer());
            }
        }
        Some(values)
    }

    #[test]
    fn calls_are_recorded_when_seen_again() {
        let cache = FunctionCallCache::default();
        assert!(matches!(cache.lookup(key(1)), CallLookup::Execute));
        record(&cache, key(1), 150, true);
        assert_eq!(cached_rows(&cache, key(1)), Some((0..150).collect()));

        assert!(matches!(cache.lookup(key(2)), CallLookup::Execute));
        record(&cache, key(2), 0, true);
        assert_eq!(cached_rows(&cache, key(2)), Some(vec![]));
    }

    #[test]
    fn incomplete_calls_are_not_cached() {
        let cache = FunctionCallCache::default();
        assert!(matches!(cache.lookup(key(1)), CallLookup::Execute));
        record(&cache, key(1), 1, false);
        assert_eq!(cached_rows(&cache, key(1)), None);
        record(&cache, key(1), 2, true);
        assert_eq!(cached_rows(&cache, key(1)), Some(vec![0, 1]));
    }

    #[test]
    fn calls_above_the_row_limits_are_not_cached() {
        let cache = FunctionCallCache::default();
        assert!(matches!(cache.lookup(key(0)), CallLookup::Execute));
        record(&cache, key(0), FUNCTION_CALL_CACHE_ROWS_PER_CALL_MAX as i64 + 1, true);
        assert!(matches!(cache.lookup(key(0)), CallLookup::Execute));

        // the cache holds the rows of eight calls of the largest size
        for argument in 1..=9 {
            assert!(matches!(cache.lookup(key(argument)), CallLookup::Execute));
        }
        for argument in 1..=8 {
            record(&cache, key(argument), FUNCTION_CALL_CACHE_ROWS_PER_CALL_MAX as i64, true);
        }
        assert!(matches!(cache.lookup(key(9)), CallLookup::Execute));
        assert!(cached_rows(&cache, key(8)).is_some());
    }
}
