use super::super::checkpoint_owner::{
    Budget, Consumed, OwnerEnvelope, Parent, ProcessIdentity, Release, ResumeRequest, Stop,
    Terminal,
};
use super::super::checkpoint_state::{CheckpointRef, Identity};
use super::super::geometry::EVIDENCE_RULE;
use super::super::state::State;
use super::super::storage_paths as paths;
use anyhow::Result;
use serde::Serialize;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT: AtomicU64 = AtomicU64::new(0);
static NAMESPACE: OnceLock<String> = OnceLock::new();

fn namespace() -> &'static str {
    NAMESPACE.get_or_init(|| {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("work/causal");
        fs::create_dir_all(&base).expect("create persistence fixture namespace parent");
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("fixture clock precedes Unix epoch")
            .as_nanos();
        for collision in 0..1024 {
            let key = format!(
                "{}-{started}-{}-{collision}",
                std::process::id(),
                module_path!()
            );
            let id = format!(
                "persistence-fixture-{}",
                &paths::digest(key.as_bytes())[..32]
            );
            match fs::create_dir(base.join(&id)) {
                Ok(()) => return id,
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("reserve persistence fixture namespace: {error}"),
            }
        }
        panic!("persistence fixture namespace collision limit exceeded");
    })
}

pub(super) struct Fixture {
    pub workspace: PathBuf,
    pub root: PathBuf,
}

impl Fixture {
    pub fn new() -> Result<Self> {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).canonicalize()?;
        let parent = workspace.join("work/persistence-tests");
        fs::create_dir_all(&parent)?;
        let root = parent.join(format!(
            "{}-{}",
            namespace(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root)?;
        Ok(Self { workspace, root })
    }
    pub fn attempt(&self, id: &str) -> Result<PathBuf> {
        let path = self.root.join(id);
        fs::create_dir(&path)?;
        Ok(path)
    }
}

pub(super) fn identity(state: &State) -> Result<Identity> {
    Ok(Identity {
        schema: 1,
        protocol: super::super::config::PROTOCOL.into(),
        protocol_sha256: super::super::config::PROTOCOL_SHA256.into(),
        config_sha256: "b".repeat(64),
        source_sha256: "c".repeat(64),
        runtime_sha256: "d".repeat(64),
        kernel: crate::ternary::kernel_name()?.into(),
        geometry: state.geometry,
        data: state.cursor.data.clone(),
        evidence_rule: EVIDENCE_RULE.into(),
        rng: "splitmix64-midpoint23-v1".into(),
    })
}

pub(super) fn owner(state: &State, id: &str) -> OwnerEnvelope {
    OwnerEnvelope {
        schema: 1,
        run_id: namespace().into(),
        attempt_id: id.into(),
        parent: None,
        budget: Budget {
            maximum_opportunities: 128,
            maximum_wall_millis: 60_000,
            maximum_attempts: 4,
            disk_bytes_per_attempt: 8 * 1024 * 1024,
        },
        consumed: Consumed {
            opportunities: state.global_opportunity,
            wall_millis: state.global_opportunity + 1,
            attempts: 1,
        },
        attempt_start_opportunity: 0,
        attempt_start_wall_millis: 0,
        observer_sequence: state.global_opportunity,
        stop: Stop::Running,
        physical_work_unknown: false,
    }
}

pub(super) fn write_json(path: &Path, value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(paths::digest(&bytes))
}

pub(super) fn release(
    parent: &Path,
    state: &State,
    old: &OwnerEnvelope,
    checkpoint: &CheckpointRef,
    next: &str,
    status: Terminal,
) -> Result<ResumeRequest> {
    let mut consumed = old.consumed.clone();
    consumed.wall_millis += 2;
    let release = Release {
        schema: 1,
        run_id: old.run_id.clone(),
        attempt_id: old.attempt_id.clone(),
        checkpoint: checkpoint.clone(),
        status,
        consumed: consumed.clone(),
        observer_sequence: old.observer_sequence + 1,
        child_reaped: true,
        observer_acknowledged: true,
        physical_work_unknown: false,
        owner_identity: ProcessIdentity {
            pid: 1,
            creation_filetime: NEXT.fetch_add(1, Ordering::Relaxed) + 1,
        },
    };
    let hash = write_json(&parent.join("release.json"), &release)?;
    write_json(
        &parent.join("release-ready.json"),
        &json!({"schema":1,"release_sha256":hash}),
    )?;
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).canonicalize()?;
    let claims = workspace
        .join("work/causal")
        .join(&old.run_id)
        .join("claims");
    fs::create_dir_all(&claims)?;
    write_json(
        &claims.join(format!("{hash}.json")),
        &json!({"schema":1,"release_sha256":hash,"new_attempt_id":next}),
    )?;
    let identity =
        super::super::checkpoint::inspect(&parent.join("persistence"), checkpoint)?.identity;
    let seal_hash = write_json(
        &parent.join("release-seal.json"),
        &json!({"schema":1,
        "release_sha256":hash,"owner_identity":release.owner_identity,"owner_reaped":true,
        "owner_exit_code":0,"external_wall_millis":release.consumed.wall_millis-old.attempt_start_wall_millis,
        "source_sha256":identity.source_sha256,"runtime_sha256":identity.runtime_sha256,
        "finalizer_identity":{"pid":2,"creation_filetime":1}}),
    )?;
    let mut new = old.clone();
    new.attempt_id = next.into();
    new.parent = Some(Parent {
        attempt_id: old.attempt_id.clone(),
        checkpoint_sha256: checkpoint.manifest_sha256.clone(),
    });
    new.consumed = consumed;
    new.consumed.attempts += 1;
    new.attempt_start_opportunity = state.global_opportunity;
    new.attempt_start_wall_millis = new.consumed.wall_millis;
    new.observer_sequence = release.observer_sequence + 1;
    new.stop = Stop::Running;
    Ok(ResumeRequest {
        release_sha256: hash,
        release_seal_sha256: seal_hash,
        explicit_user_renewal: false,
        new_owner: new,
    })
}

pub(super) fn parent_hashes(parent: &Path) -> Result<Vec<(String, String)>> {
    let mut result = Vec::new();
    for entry in fs::read_dir(parent)? {
        let path = entry?.path();
        if path.is_dir() {
            for (name, _) in paths::inventory(&path)? {
                result.push((
                    format!("persistence/{name}"),
                    paths::digest(&paths::read(&path, &name, paths::FILE_CAP)?),
                ));
            }
        } else {
            result.push((
                path.file_name().unwrap().to_string_lossy().into_owned(),
                paths::digest(&fs::read(path)?),
            ));
        }
    }
    result.sort_unstable();
    Ok(result)
}
