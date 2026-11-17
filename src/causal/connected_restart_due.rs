//! Which due tiles restart on an anchor, and how long waiting tiles have been due.
//! See CONNECTED.md §Restart options (stagger).

use super::connected_restart::TileRestart;
use super::connected_restart_config::{Law, Restart};
use super::geometry::TILES;

/// Whether `entry` is due under a discrete-restart `config`.
fn is_due(entry: &TileRestart, config: Restart) -> bool {
    config.law == Law::Restart && config.threshold.is_some() && entry.drift >= entry.bar
}

/// Tiles restarting now: every due tile, or under `stagger` at most one once the window passed,
/// the longest-waiting first (ties: larger drift / bar, then lowest id).
pub(super) fn due(tiles: &[TileRestart], config: Restart, anchors: u64) -> Vec<usize> {
    let due = (0..TILES).filter(|&tile| is_due(&tiles[tile], config));
    if config.stagger == 0 {
        return due.collect();
    }
    // The last restart's anchor is the newest origin of any tile that has restarted.
    let last = tiles
        .iter()
        .filter(|e| e.restarts > 0)
        .map(|e| e.origin)
        .max();
    if last.is_some_and(|last| anchors - last < config.stagger) {
        return Vec::new();
    }
    let since = |tile: usize| tiles[tile].due_since.unwrap_or(anchors);
    let ratio = |tile: usize| tiles[tile].drift / tiles[tile].bar;
    due.min_by(|&a, &b| {
        since(a)
            .cmp(&since(b))
            .then(ratio(b).total_cmp(&ratio(a)))
            .then(a.cmp(&b))
    })
    .into_iter()
    .collect()
}

/// After restarts: a tile still due keeps (or starts) its due-since anchor; others clear it.
pub(super) fn mark_waiting(tiles: &mut [TileRestart], config: Restart, anchors: u64) {
    for entry in tiles {
        entry.due_since = if is_due(entry, config) {
            Some(entry.due_since.unwrap_or(anchors))
        } else {
            None
        };
    }
}

/// A tile waits exactly while due; it cannot have become due after `anchors`.
pub(super) fn consistent(entry: &TileRestart, config: Restart, anchors: u64) -> bool {
    entry.due_since.is_some() == is_due(entry, config)
        && entry.due_since.is_none_or(|since| since <= anchors)
}

#[cfg(test)]
mod tests {
    use super::{due, Restart, TileRestart};

    #[test]
    fn stagger_waits_out_the_window_then_serves_the_longest_due() {
        let config = |stagger| Restart {
            threshold: Some(1.0),
            stagger,
            ..Restart::default()
        };
        // Tile 0 restarted at anchor 5; tile 1 has waited since 2 at ratio 1.1, tile 3 is
        // newly due at ratio 3 (or has also waited since 2).
        let tiles = |older: Option<u64>| {
            let mut tiles = vec![TileRestart::new(1.0, 1.0); 7];
            (tiles[0].restarts, tiles[0].origin) = (1, 5);
            (tiles[1].drift, tiles[1].due_since) = (1.1, Some(2));
            (tiles[3].drift, tiles[3].due_since) = (3.0, older);
            tiles
        };
        // (stagger, anchors, tile 3's due-since, chosen)
        for (stagger, anchors, older, chosen) in [
            (4, 10, None, vec![1]),
            // Exactly `stagger` anchors after the last restart the window has passed.
            (4, 9, None, vec![1]),
            (4, 8, None, vec![]),
            // Equal waits: the larger drift / bar ratio.
            (4, 10, Some(2), vec![3]),
            (0, 10, None, vec![1, 3]),
        ] {
            let found = due(&tiles(older), config(stagger), anchors);
            assert_eq!(found, chosen, "{stagger} {anchors} {older:?}");
        }
    }
}
