use super::ProcessIdentity;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type Handle = *mut c_void;
#[repr(C)]
#[derive(Default)]
struct FileTime {
    low: u32,
    high: u32,
}

#[link(name = "kernel32")]
extern "system" {
    fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
    fn CloseHandle(handle: Handle) -> i32;
    fn GetCurrentProcess() -> Handle;
    fn IsProcessInJob(process: Handle, job: Handle, result: *mut i32) -> i32;
    fn GetProcessTimes(
        process: Handle,
        creation: *mut FileTime,
        exit: *mut FileTime,
        kernel: *mut FileTime,
        user: *mut FileTime,
    ) -> i32;
    fn GetExitCodeProcess(process: Handle, code: *mut u32) -> i32;
}

pub(crate) struct OwnerProcess(Handle);
impl Drop for OwnerProcess {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

impl OwnerProcess {
    fn open(identity: ProcessIdentity) -> Result<Self> {
        ensure!(
            identity.pid > 0 && identity.creation_filetime > 0,
            "invalid launch process identity"
        );
        let handle = unsafe { OpenProcess(0x1000, 0, identity.pid) };
        ensure!(!handle.is_null(), "cannot inspect owned process");
        let process = Self(handle);
        let (mut creation, mut exit, mut kernel, mut user) = (
            FileTime::default(),
            FileTime::default(),
            FileTime::default(),
            FileTime::default(),
        );
        ensure!(
            unsafe { GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) }
                != 0,
            "cannot read owned process creation identity"
        );
        let actual = (u64::from(creation.high) << 32) | u64::from(creation.low);
        ensure!(
            actual == identity.creation_filetime,
            "owned process PID was reused"
        );
        process.alive()?;
        Ok(process)
    }

    pub fn alive(&self) -> Result<()> {
        let mut code = 0u32;
        ensure!(
            unsafe { GetExitCodeProcess(self.0, &mut code) } != 0 && code == 259,
            "owner process is no longer alive"
        );
        Ok(())
    }
}

pub(super) fn verify(owner: ProcessIdentity, child: ProcessIdentity) -> Result<OwnerProcess> {
    let mut in_job = 0i32;
    ensure!(
        unsafe { IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut in_job) } != 0
            && in_job != 0,
        "worker lacks assigned Windows Job"
    );
    let owner = OwnerProcess::open(owner)?;
    let _child = OwnerProcess::open(child)?;
    Ok(owner)
}
