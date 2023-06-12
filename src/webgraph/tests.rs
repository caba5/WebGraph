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

#[test]
fn test_iteration_on_successors_of_first_node() {
    let mut graph = build_graph();
    let mut succ_it = graph.successors(0).unwrap();

    assert_eq!(succ_it.next().unwrap(), 1);
    assert_eq!(succ_it.next().unwrap(), 4);
    assert_eq!(succ_it.next().unwrap(), 5);
    assert_eq!(succ_it.next().unwrap(), 6);
    assert_eq!(succ_it.next().unwrap(), 7);
    assert_eq!(succ_it.next().unwrap(), 8);
    assert_eq!(succ_it.next().unwrap(), 13);
    assert_eq!(succ_it.next().unwrap(), 38);
    assert_eq!(succ_it.next().unwrap(), 48);
    assert_eq!(succ_it.next().unwrap(), 154);
    assert_eq!(succ_it.next(), None);
}

#[test]
fn test_iteration_on_successors_of_mid_node() {
    let mut graph = build_graph();
    let mut succ_it = graph.successors(1).unwrap();

    assert_eq!(succ_it.next().unwrap(), 10);
    assert_eq!(succ_it.next().unwrap(), 38);
    assert_eq!(succ_it.next().unwrap(), 42);
    assert_eq!(succ_it.next().unwrap(), 46);
    assert_eq!(succ_it.next().unwrap(), 49);
    assert_eq!(succ_it.next().unwrap(), 50);
    assert_eq!(succ_it.next().unwrap(), 51);
    assert_eq!(succ_it.next(), None);
}

#[test]
fn test_iteration_on_successors_of_last_node() {
    let mut graph = build_graph();
    let last_node = graph.num_nodes() - 1;
    let mut succ_it = graph.successors(last_node).unwrap();

    assert_eq!(succ_it.next().unwrap(), 99979);
    assert_eq!(succ_it.next().unwrap(), 99980);
    assert_eq!(succ_it.next().unwrap(), 99981);
    assert_eq!(succ_it.next().unwrap(), 99982);
    assert_eq!(succ_it.next().unwrap(), 99983);
    assert_eq!(succ_it.next().unwrap(), 99984);
    assert_eq!(succ_it.next().unwrap(), 99985);
    assert_eq!(succ_it.next().unwrap(), 99986);
    assert_eq!(succ_it.next().unwrap(), 99987);
    assert_eq!(succ_it.next().unwrap(), 99988);
    assert_eq!(succ_it.next().unwrap(), 99989);
    assert_eq!(succ_it.next().unwrap(), 99990);
    assert_eq!(succ_it.next().unwrap(), 99998);
    assert_eq!(succ_it.next().unwrap(), 99999);
    assert_eq!(succ_it.next(), None);
}

#[test]
fn test_iteration_on_successors_out_of_bounds() {
    let mut graph = build_graph();
    let succ_it = graph.successors(graph.num_nodes());
    
    assert!(succ_it.is_none());
}