use crate::uncompressed_graph::{ImmutableGraphBuilder, self};

use super::*;

fn build_graph() -> BVGraphPlain {
    BVGraphPlainBuilder::new()
        .load_graph("new_clear")
        .load_offsets("new_clear")
        .build()
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
    let graph = build_graph();
    let succ_it = graph.successors_plain(0).unwrap();

    let correct = vec![1, 4, 5, 6, 7, 8, 13, 38, 48, 154].into_iter();

    assert!(succ_it.eq(correct));
}

#[test]
fn test_iteration_on_successors_of_mid_node() {
    let graph = build_graph();
    let succ_it = graph.successors_plain(1).unwrap();

    let correct = vec![10, 38, 42, 46, 49, 50, 51].into_iter();

    assert!(succ_it.eq(correct));
}

#[test]
fn test_iteration_on_successors_of_last_node() {
    let graph = build_graph();
    let last_node = graph.num_nodes() - 1;
    let succ_it = graph.successors_plain(last_node).unwrap();

    let correct = 
        vec![99979, 99980, 99981, 99982, 99983, 99984, 99985, 99986, 99987, 99988, 99989, 99990, 99998, 99999].into_iter();

    assert!(succ_it.eq(correct));
}

#[test]
fn test_iteration_on_successors_out_of_bounds() {
    let graph = build_graph();
    let succ_it = graph.successors_plain(graph.num_nodes());
    
    assert!(succ_it.is_none());
}

#[test]
fn test_build_from_uncompressed_graph() {
    let uncompressed_graph = ImmutableGraphBuilder::<usize>::new()
                                .load_graph("clear")
                                .load_offsets("clear")
                                .count_nodes()
                                .count_arcs()
                                .build();
    let uncompressed_graph_copy = uncompressed_graph.clone();
    let webgraph = BVGraphPlainBuilder::from(uncompressed_graph).build();

    let correct_webgraph = build_graph();

    assert_eq!(webgraph.num_nodes(), uncompressed_graph_copy.num_nodes());
    assert_eq!(webgraph.num_arcs(), uncompressed_graph_copy.num_arcs());

    assert_eq!(webgraph.num_nodes(), correct_webgraph.num_nodes());
    assert_eq!(webgraph.num_arcs(), correct_webgraph.num_arcs());
    assert_eq!(webgraph.offsets, correct_webgraph.offsets);
    assert_eq!(webgraph.graph_memory, correct_webgraph.graph_memory);
}