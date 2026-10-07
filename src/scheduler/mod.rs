mod costs;
mod deadline;

pub(crate) use costs::CostModel;
pub(crate) use deadline::select_batch;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum BatchKind {
    Open,
    Close,
    Discard,
    Activate,
    Prefill,
    Decode,
}

#[cfg(test)]
mod tests;
