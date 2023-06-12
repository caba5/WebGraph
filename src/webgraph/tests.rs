use super::*;

fn build_graph() -> BVGraph {
    BVGraphBuilder::new()
                            .load_graph("new_clear")
                            .load_offsets("new_clear")
                            .construct()
}

#[test]
fn test_graph_building_nodes() {
    let graph = build_graph();
    
    assert_eq!(graph.num_nodes(), 100000);
}

#[test]
fn test_graph_building_arcs() {
    let graph = build_graph();
    
    assert_eq!(graph.num_arcs(), 3050615);
}

#[test]
fn test_outdegree_of_first_node() {
    let mut graph = build_graph();
    
    assert_eq!(graph.outdegree(0).unwrap(), 10);
}

#[test]
fn test_outdegree_of_mid_node() {
    let mut graph = build_graph();
    
    assert_eq!(graph.outdegree(1).unwrap(), 7);
}

#[test]
fn test_outdegree_of_last_node() {
    let mut graph = build_graph();
    let last_node = graph.num_nodes() - 1;

    assert_eq!(graph.outdegree(last_node).unwrap(), 14);
}

#[test]
fn test_outdegree_out_of_bounds() {
    let mut graph = build_graph();

    assert_eq!(graph.outdegree(graph.num_nodes()), None);
}