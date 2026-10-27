use super::math::fingerprint;
use super::node::Node;
use super::training_pulse::TrainingPulse;
use super::{TILES, WIDTH};
use crate::ternary::TernaryMatrix;
use crate::topology::Mesh;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredNode {
    schema: u32,
    id: usize,
    version: u64,
    transform: Vec<u8>,
    curve: Vec<u8>,
    router: Vec<u8>,
    pulse_context: Vec<u8>,
    pulse_heads: Vec<u8>,
    checksum: u64,
}

#[derive(Default, Serialize)]
pub(super) struct IoCounts {
    pub bytes_read: usize,
    pub bytes_written: usize,
    pub loads: usize,
    pub commits: usize,
    pub initialization_bytes_written: usize,
    pub peak_live_tiles: usize,
    pub peak_serialized_tile_bytes: usize,
}

pub(super) struct Store {
    pub directory: PathBuf,
    pub versions: [u64; TILES],
    pub io: IoCounts,
    pub mesh: Mesh,
    busy: bool,
}

impl Store {
    pub fn create(directory: &Path, seed: u64) -> Result<Self> {
        let root = std::env::current_dir()?.canonicalize()?;
        let owned = root.join("work");
        fs::create_dir_all(&owned)?;
        let owned = owned.canonicalize()?;
        ensure!(
            owned.starts_with(&root) && owned != root,
            "work directory escapes the workspace"
        );
        let parent = directory
            .parent()
            .context("missing run parent")?
            .canonicalize()?;
        ensure!(
            parent.starts_with(&owned),
            "text artifacts must be under current workspace/work"
        );
        ensure!(!directory.exists(), "text run directory must be fresh");
        fs::create_dir(directory)?;
        let mut store = Self {
            directory: directory.canonicalize()?,
            versions: [0; TILES],
            io: IoCounts::default(),
            mesh: Mesh::new(1)?,
            busy: false,
        };
        for id in 0..TILES {
            let node = Node::new(
                seed.wrapping_add(100 + id as u64 * 5),
                store.mesh.neighbors(id)?.len() + 1,
            )?;
            store.save(id, &node)?;
        }
        store.io.initialization_bytes_written = store.io.bytes_written;
        store.io.peak_live_tiles = 1;
        Ok(store)
    }

    /// Scope the sole decoded tile to a closure; traces never contain weights.
    pub fn with_node<T>(
        &mut self,
        id: usize,
        operation: impl FnOnce(&mut Node) -> Result<T>,
    ) -> Result<T> {
        ensure!(id < TILES && !self.busy, "invalid or nested tile residency");
        self.busy = true;
        let result = self.operate(id, operation);
        self.busy = false;
        result
    }

    fn operate<T>(
        &mut self,
        id: usize,
        operation: impl FnOnce(&mut Node) -> Result<T>,
    ) -> Result<T> {
        let mut node = self.load(id)?;
        let prior = node.version;
        let result = operation(&mut node)?;
        ensure!(
            node.version == prior || node.version == prior + 1,
            "invalid tile version transition"
        );
        if node.version != prior {
            self.save(id, &node)?;
            self.versions[id] = node.version;
        }
        Ok(result)
    }

    fn load(&mut self, id: usize) -> Result<Node> {
        let mut bytes = Vec::new();
        File::open(self.path(id, self.versions[id]))?
            .take(8193)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 8192, "tile artifact exceeds bound");
        let stored: StoredNode = serde_json::from_slice(&bytes)?;
        ensure!(
            stored.schema == 2 && stored.id == id && stored.version == self.versions[id],
            "tile identity mismatch"
        );
        ensure!(
            stored.checksum == stored.checksum_value(),
            "tile checksum mismatch"
        );
        let node = Node {
            transform: TernaryMatrix::from_packed(WIDTH, WIDTH, &stored.transform)?,
            curve: TernaryMatrix::from_packed(3, WIDTH, &stored.curve)?,
            router: TernaryMatrix::from_packed(
                self.mesh.neighbors(id)?.len() + 1,
                WIDTH * 2,
                &stored.router,
            )?,
            pulse: TrainingPulse {
                context: TernaryMatrix::from_packed(WIDTH, 2 * WIDTH, &stored.pulse_context)?,
                heads: TernaryMatrix::from_packed(
                    2 * WIDTH + self.mesh.neighbors(id)?.len() + 1,
                    WIDTH,
                    &stored.pulse_heads,
                )?,
            },
            version: stored.version,
        };
        self.io.bytes_read += bytes.len();
        self.io.loads += 1;
        self.io.peak_serialized_tile_bytes = self.io.peak_serialized_tile_bytes.max(bytes.len());
        Ok(node)
    }

    fn save(&mut self, id: usize, node: &Node) -> Result<()> {
        let mut stored = StoredNode {
            schema: 2,
            id,
            version: node.version,
            transform: node.transform.packed_bytes().to_vec(),
            curve: node.curve.packed_bytes().to_vec(),
            router: node.router.packed_bytes().to_vec(),
            pulse_context: node.pulse.context.packed_bytes().to_vec(),
            pulse_heads: node.pulse.heads.packed_bytes().to_vec(),
            checksum: 0,
        };
        stored.checksum = stored.checksum_value();
        let bytes = serde_json::to_vec(&stored)?;
        ensure!(bytes.len() <= 8192, "serialized tile exceeds bound");
        write_new(&self.path(id, node.version), &bytes)?;
        self.io.bytes_written += bytes.len();
        self.io.commits += 1;
        self.io.peak_serialized_tile_bytes = self.io.peak_serialized_tile_bytes.max(bytes.len());
        Ok(())
    }

    fn path(&self, id: usize, version: u64) -> PathBuf {
        self.directory.join(format!("tile-{id}-v{version}.json"))
    }
}

impl StoredNode {
    fn checksum_value(&self) -> u64 {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.schema.to_le_bytes());
        bytes.extend_from_slice(&(self.id as u64).to_le_bytes());
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.transform);
        bytes.extend_from_slice(&self.curve);
        bytes.extend_from_slice(&self.router);
        bytes.extend_from_slice(&self.pulse_context);
        bytes.extend_from_slice(&self.pulse_heads);
        fingerprint(&bytes)
    }
}

pub(super) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension("json.pending");
    ensure!(!path.exists(), "refusing to replace an artifact");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_roundtrip_and_corruption_fail_closed() -> Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let directory = std::env::current_dir()?
            .join("work")
            .join(format!("text-storage-test-{}-{unique}", std::process::id()));
        let mut store = Store::create(&directory, 53)?;
        let expected = Node::new(153, store.mesh.neighbors(0)?.len() + 1)?;
        store.with_node(0, |loaded| {
            assert_eq!(loaded.transform, expected.transform);
            assert_eq!(loaded.curve, expected.curve);
            assert_eq!(loaded.router, expected.router);
            assert_eq!(loaded.pulse.context, expected.pulse.context);
            assert_eq!(loaded.pulse.heads, expected.pulse.heads);
            Ok(())
        })?;
        let path = store.path(0, 0);
        let mut stored: StoredNode = serde_json::from_slice(&fs::read(&path)?)?;
        stored.pulse_context[0] ^= 1;
        fs::write(path, serde_json::to_vec(&stored)?)?;
        assert!(store.with_node(0, |_| Ok(())).is_err());
        assert_eq!(store.io.peak_live_tiles, 1);
        Ok(())
    }
}
