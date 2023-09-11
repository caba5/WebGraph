use crate::uncompressed_graph::UncompressedGraphBuilder;

use super::*;

const TEST_DATA_PATH: &str = "src/plain_webgraph/test_data/";

fn build_graph() -> BVGraphPlain {
    BVGraphPlainBuilder::new()
        .load_graph_uncompressed(&(TEST_DATA_PATH.to_owned() + "webgraph"))
        .load_offsets_uncompressed(&(TEST_DATA_PATH.to_owned() + "webgraph"))
        .count_nodes()
        .count_edges()
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
    let uncompressed_graph = UncompressedGraphBuilder::<usize>::new()
                                .load_graph(&(TEST_DATA_PATH.to_owned() + "uncompressed"))
                                .load_offsets(&(TEST_DATA_PATH.to_owned() + "uncompressed"))
                                .count_nodes()
                                .count_edges()
                                .build();

    let uncompressed_num_nodes = uncompressed_graph.num_nodes();
    let uncompressed_num_arcs = uncompressed_graph.num_arcs();

    let webgraph = BVGraphPlainBuilder::from(uncompressed_graph).build();

    let correct_webgraph = build_graph();

    assert_eq!(webgraph.num_nodes(), uncompressed_num_nodes);
    assert_eq!(webgraph.num_arcs(), uncompressed_num_arcs);

    assert_eq!(webgraph.num_nodes(), correct_webgraph.num_nodes());
    assert_eq!(webgraph.num_arcs(), correct_webgraph.num_arcs());
    assert_eq!(webgraph.offsets, correct_webgraph.offsets);
    assert_eq!(webgraph.graph_memory, correct_webgraph.graph_memory);
}

#[test]
fn test_converting_iteration_of_first_node() {
    let graph = build_graph();

    let mut iter = graph.conversion_iterator_from(0);

    let correct_successors = [1, 4, 5, 6, 7, 8, 13, 38, 48, 154];

    assert_eq!(iter.next().unwrap(), 0);
    assert_eq!(iter.outdegree(), 10);
    assert_eq!(iter.successor_array(), correct_successors);
}

#[test]
fn test_converting_iteration_of_mid_node() {
    let graph = build_graph();

    let mid_node = 1;

    let mut iter = graph.conversion_iterator_from(mid_node);

    let correct_successors = [10, 38, 42, 46, 49, 50, 51];

    assert_eq!(iter.next().unwrap(), mid_node);
    assert_eq!(iter.outdegree(), 7);
    assert_eq!(iter.successor_array(), correct_successors);
}

#[test]
fn test_converting_iteration_of_last_node() {
    let graph = build_graph();

    let last_node = graph.num_nodes() - 1;

    let mut iter = graph.conversion_iterator_from(last_node);

    let correct_successors = 
        [99979, 99980, 99981, 99982, 99983, 99984, 99985, 99986, 99987, 99988, 99989, 99990, 99998, 99999];

    assert_eq!(iter.next().unwrap(), last_node);
    assert_eq!(iter.outdegree(), 14);
    assert_eq!(iter.successor_array(), correct_successors);
}

#[test]
fn test_converting_iteration_out_of_bounds() {
    let graph = build_graph();
    let mut iter = graph.conversion_iterator_from(graph.num_nodes());
    
    assert_eq!(iter.next(), None);
}