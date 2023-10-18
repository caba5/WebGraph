use num_traits::ToPrimitive;

use super::*;

const TEST_DATA_PATH: &str = "src/ascii_graph/test_data/";
const TEST_FILE: &str = "cnr-2000-ascii";
const TEST_PERMUTED_FILE: &str = "cnr-2000-permuted-ascii";

#[test]
fn test_graph_building_ascii() {
    let uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .build();
    assert_eq!(uncompressed_graph.num_nodes(), 325557);
    assert_eq!(uncompressed_graph.num_arcs(), 3216152);
}

#[test]
fn test_graph_building_bin() {
    let uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_graph_bin(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .load_offsets_bin(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .count_nodes()
        .count_edges()
        .build();
    assert_eq!(uncompressed_graph.num_nodes(), 325557);
    assert_eq!(uncompressed_graph.num_arcs(), 3216152);
}

#[test]
fn test_iterate_on_first_successors() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_graph_bin(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .load_offsets_bin(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .count_nodes()
        .count_edges()
        .build();
    let it = uncompressed_graph.successors(0);

    let correct = vec![1, 4, 8, 219, 220];

    assert!(it.eq(correct));
}

#[test]
fn test_iterate_on_mid_successors() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_graph_bin(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .load_offsets_bin(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .count_nodes()
        .count_edges()
        .build();
    let it = uncompressed_graph.successors(6);

    let correct = vec![5, 7, 8, 219, 220].into_iter();

    assert!(it.eq(correct));
}

#[test]
fn test_iterate_on_end_successors() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_graph_bin(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .load_offsets_bin(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .count_nodes()
        .count_edges()
        .build();
    let last_node = uncompressed_graph.n - 1;  // n-1 since it is the last node
    let it = uncompressed_graph.successors(last_node);

    let correct = 
        vec![289276, 289277, 289278, 289279, 289280, 325555].into_iter();

    assert!(it.eq(correct));
}

#[test]
fn test_from_ascii_first_index() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .build();

    let it = uncompressed_graph.successors(0);

    let correct = vec![1, 4, 8, 219, 220];

    assert!(it.eq(correct));
}

#[test]
fn test_from_ascii_middle_index() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .build();

    let it = uncompressed_graph.successors(181145);

    let correct = vec![181146, 181147, 181148, 181149, 181150, 181151, 181152, 181153, 181154];

    assert!(it.eq(correct));
}

#[test]
fn test_from_ascii_last_index() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
        .build();

    let it = uncompressed_graph.successors(325556);

    let correct = vec![289276, 289277, 289278, 289279, 289280, 325555];

    assert!(it.eq(correct));
}

#[test]
fn test_from_permuted_ascii_first_node() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_PERMUTED_FILE))
        .build();

    let it = uncompressed_graph.successors(0);

    let correct = vec![1, 4, 8, 219, 220];

    assert!(it.eq(correct));
}

#[test]
fn test_from_permuted_ascii_middle_node() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_PERMUTED_FILE))
        .build();

    let it = uncompressed_graph.successors(181145);

    let correct = vec![181146, 181147, 181148, 181149, 181150, 181151, 181152, 181153, 181154];

    assert!(it.eq(correct));
}

#[test]
fn test_from_permuted_ascii_last_node() {
    let mut uncompressed_graph: AsciiGraph<usize> = 
        AsciiGraphBuilder::new()
        .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_PERMUTED_FILE))
        .build();

    let it = uncompressed_graph.successors(325556);

    let correct = vec![289276, 289277, 289278, 289279, 289280, 325555];

    assert!(it.eq(correct));
}

#[test]
fn test_bvgraph_iteration() {
    let uncompressed_graph: AsciiGraph<usize> = 
    AsciiGraphBuilder::new()
    .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_FILE))
    .build();

    let mut it = uncompressed_graph.iter();

    let node = it.next().unwrap().to_usize().unwrap();
    let outd = it.next().unwrap().to_usize().unwrap();
    let mut succs = Vec::with_capacity(outd);

    for _ in 0..outd {
        succs.push(it.next().unwrap().to_usize().unwrap());
    }

    assert_eq!(node, 0);
    assert_eq!(outd, 5);
    assert_eq!(succs.as_slice(), [1, 4, 8, 219, 220]);

    let node = it.next().unwrap().to_usize().unwrap();
    let outd = it.next().unwrap().to_usize().unwrap();
    let mut succs = Vec::with_capacity(outd);

    for _ in 0..outd {
        succs.push(it.next().unwrap().to_usize().unwrap());
    }

    assert_eq!(node, 1);
    assert_eq!(outd, 5);
    assert_eq!(succs.as_slice(), [0, 7, 8, 219, 220]);
}

#[test]
fn test_bvgraph_iteration_permuted() {
    let uncompressed_graph: AsciiGraph<usize> = 
    AsciiGraphBuilder::new()
    .load_ascii(&(TEST_DATA_PATH.to_owned() + TEST_PERMUTED_FILE))
    .build();

    let mut it = uncompressed_graph.iter();

    let node = it.next().unwrap().to_usize().unwrap();
    let outd = it.next().unwrap().to_usize().unwrap();
    let mut succs = Vec::with_capacity(outd);

    for _ in 0..outd {
        succs.push(it.next().unwrap().to_usize().unwrap());
    }

    assert_eq!(node, 20758);
    assert_eq!(outd, 10);
    assert_eq!(succs.as_slice(), [20735, 20736, 20737, 20738, 20740, 20752, 20754, 20755, 20756, 272816]);

    let node = it.next().unwrap().to_usize().unwrap();
    let outd = it.next().unwrap().to_usize().unwrap();
    let mut succs = Vec::with_capacity(outd);

    for _ in 0..outd {
        succs.push(it.next().unwrap().to_usize().unwrap());
    }

    assert_eq!(node, 23317);
    assert_eq!(outd, 3);
    assert_eq!(succs.as_slice(), [23315, 23316, 272816]);
}