use super::*;

#[test]
fn test_properties_serialization() {
    let graph = BVGraph::new();
    graph.store("test");

    let loaded_graph = BVGraph::load("test");

    assert_eq!(graph, loaded_graph);
}