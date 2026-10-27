use hexweave::{
    config::Config,
    planner::{plan, Action, ExitReason},
    topology::{Axial, Mesh},
};

#[test]
fn profiles_have_checked_main_matrix_counts() {
    let config = Config::load("configs/local-287m.json".as_ref()).unwrap();
    let estimate = config.estimate().unwrap();
    assert_eq!(estimate.tiles, 37);
    assert_eq!(estimate.main_parameters, 287_047_680);
    assert_eq!(estimate.fp32_adam_main_state_floor_bytes, 4_592_762_880);
    assert_eq!(estimate.overhead_parameters, None);
    assert!(!estimate.training_admitted);
    assert!(!estimate.executable);
    let development = Config::load("configs/dev-81m.json".as_ref()).unwrap();
    assert_eq!(development.estimate().unwrap().main_parameters, 81_526_784);
}

#[test]
fn geometry_and_routes_are_bounded_before_allocation() {
    let mut config = Config::smoke();
    for radius in [17, usize::MAX] {
        config.radius = radius;
        assert!(config.estimate().is_err());
        assert!(Mesh::new(radius).is_err());
    }
    config = Config::smoke();
    config.hidden = usize::MAX;
    assert!(config.estimate().is_err());
    config = Config::smoke();
    config.heads = 0;
    assert!(config.validate().is_err());
    config.heads = 3;
    assert!(config.validate().is_err());
    config = Config::smoke();
    config.max_depth = 0;
    assert!(config.validate().is_err());
}

#[test]
fn zero_radius_is_a_valid_singleton() {
    let mesh = Mesh::new(0).unwrap();
    assert_eq!(mesh.coordinates(), &[Axial { q: 0, r: 0 }]);
    assert!(!mesh.is_empty());
    assert!(mesh.neighbors(0).unwrap().is_empty());
    assert!(mesh.neighbors(1).is_err());
    let mut config = Config::smoke();
    config.radius = 0;
    let result = plan(&config, &mesh, 0, |_, _| panic!("policy should not run")).unwrap();
    assert_eq!(result.exits[0].reason, ExitReason::NoNeighbors);
    assert_eq!(result.exits[0].weight, 1.0);
}

#[test]
fn mesh_is_stable_bidirectional_and_has_six_neighbors_at_center() {
    let mesh = Mesh::new(3).unwrap();
    assert_eq!(mesh.coordinates(), Mesh::new(3).unwrap().coordinates());
    let center = mesh
        .coordinates()
        .iter()
        .position(|p| *p == Axial { q: 0, r: 0 })
        .unwrap();
    assert_eq!(mesh.neighbors(center).unwrap().len(), 6);
    for tile in 0..mesh.len() {
        let neighbors = mesh.neighbors(tile).unwrap();
        assert!(neighbors.len() <= 6);
        assert!(!neighbors.contains(&tile));
        for other in neighbors {
            assert!(mesh.neighbors(other).unwrap().contains(&tile));
        }
    }
}

#[test]
fn fork_plans_conserve_mass_and_respect_hard_caps() {
    let mesh = Mesh::new(1).unwrap();
    let mut config = Config::smoke();
    config.fanout = true;
    config.max_depth = 8;
    config.max_visits = 9;
    config.max_frontier = 3;
    let result = plan(&config, &mesh, 0, |_, legal| {
        Ok(Action::Fork(legal[0], legal[1]))
    })
    .unwrap();
    assert!(result.visits.len() <= config.max_visits);
    assert!(result.peak_frontier <= config.max_frontier);
    assert!((result.exits.iter().map(|exit| exit.weight).sum::<f64>() - 1.0).abs() < 1e-12);
    for visit in &result.visits {
        assert!(visit.branch.depth < config.max_depth);
        assert_eq!(visit.branch.history.len(), visit.branch.depth + 1);
        assert_eq!(visit.branch.history.last(), Some(&visit.branch.tile));
    }
}

#[test]
fn invalid_actions_fail_and_revisits_remain_legal() {
    let mesh = Mesh::new(1).unwrap();
    let mut config = Config::smoke();
    assert!(plan(&config, &mesh, 0, |_, _| Ok(Action::Single(0))).is_err());
    assert!(plan(&config, &mesh, 0, |_, legal| Ok(Action::Fork(
        legal[0], legal[1]
    )))
    .is_err());
    config.fanout = true;
    assert!(plan(&config, &mesh, 0, |_, legal| Ok(Action::Fork(
        legal[0], legal[0]
    )))
    .is_err());
    let result = plan(&config, &mesh, 0, |branch, legal| {
        Ok(Action::Single(if branch.depth == 0 { legal[0] } else { 0 }))
    })
    .unwrap();
    assert_eq!(result.visits.last().unwrap().branch.history, vec![0, 1, 0]);
}

#[test]
fn forced_exits_and_stops_preserve_single_branch_mass() {
    let mesh = Mesh::new(1).unwrap();
    let mut config = Config::smoke();
    let stopped = plan(&config, &mesh, 0, |_, _| Ok(Action::Stop)).unwrap();
    assert_eq!(stopped.exits[0].reason, ExitReason::Requested);
    config.max_visits = 1;
    config.max_frontier = 1;
    let limited = plan(&config, &mesh, 0, |_, legal| Ok(Action::Single(legal[0]))).unwrap();
    assert_eq!(limited.exits[0].reason, ExitReason::VisitLimit);
    assert_eq!(limited.exits[0].weight, 1.0);
    config.max_depth = 1;
    let depth = plan(&config, &mesh, 0, |_, _| {
        panic!("depth cap bypasses policy")
    })
    .unwrap();
    assert_eq!(depth.exits[0].reason, ExitReason::DepthLimit);
    assert!(plan(&config, &mesh, 99, |_, _| Ok(Action::Stop)).is_err());
}

#[test]
fn parsing_rejects_unknown_fields_and_negative_dimensions() {
    let mut value = serde_json::to_value(Config::smoke()).unwrap();
    value["hidden"] = (-1).into();
    assert!(serde_json::from_value::<Config>(value).is_err());
    let mut value = serde_json::to_value(Config::smoke()).unwrap();
    value["surprise"] = true.into();
    assert!(serde_json::from_value::<Config>(value).is_err());
}
