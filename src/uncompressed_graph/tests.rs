use super::*;

fn build_graph<'a, T: num_traits::PrimInt>() -> UncompressedGraph<T> 
where 
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: DeserializeOwned
{
    ImmutableGraphBuilder::<T>::new()
                            .load_graph("clear")
                            .load_offsets("clear")
                            .count_nodes()
                            .count_arcs()
                            .construct()
}

fn build_graph_bin<T: num_traits::PrimInt>() -> UncompressedGraph<T> 
where 
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: DeserializeOwned
{
    ImmutableGraphBuilder::<T>::new()
                            .load_graph_bin("clear")
                            .load_offsets_bin("clear")
                            .count_nodes()
                            .count_arcs()
                            .construct()
}

#[test]
fn test_graph_building_nodes() {
    let uncompressed_graph = build_graph::<u32>();
    assert_eq!(uncompressed_graph.num_nodes(), 100000);
}

#[test]
fn test_graph_building_arcs() {
    let uncompressed_graph = build_graph::<u32>();
    assert_eq!(uncompressed_graph.num_arcs(), 3050615);
}

#[test]
fn test_graph_building_bin_nodes() {
    let uncompressed_graph = build_graph_bin::<u32>();
    assert_eq!(uncompressed_graph.num_nodes(), 100000);
}

#[test]
fn test_iterate_on_first_successors() {
    let uncompressed_graph = build_graph_bin::<u32>();
    let mut it = uncompressed_graph.successors(0);

    assert_eq!(it.next().unwrap(), 1);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.next().unwrap(), 5);
    assert_eq!(it.next().unwrap(), 6);
    assert_eq!(it.next().unwrap(), 7);
    assert_eq!(it.next().unwrap(), 8);
    assert_eq!(it.next().unwrap(), 13);
    assert_eq!(it.next().unwrap(), 38);
    assert_eq!(it.next().unwrap(), 48);
    assert_eq!(it.next().unwrap(), 154);
    assert!(it.next().is_none());
}

#[test]
fn test_iterate_on_mid_successors() {
    let uncompressed_graph = build_graph_bin::<u32>();
    let mut it = uncompressed_graph.successors(6);

    assert_eq!(it.next().unwrap(), 0);
    assert_eq!(it.next().unwrap(), 1);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.next().unwrap(), 5);
    assert_eq!(it.next().unwrap(), 6);
    assert_eq!(it.next().unwrap(), 7);
    assert_eq!(it.next().unwrap(), 8);
    assert_eq!(it.next().unwrap(), 13);
    assert_eq!(it.next().unwrap(), 38);
    assert_eq!(it.next().unwrap(), 84821);
    assert!(it.next().is_none());
}

#[test]
fn test_iterate_on_end_successors() {
    let uncompressed_graph = build_graph_bin::<u32>();
    let last_node = (uncompressed_graph.n - 1) as u32;  // n-1 since it is the last node
    let mut it = uncompressed_graph.successors(last_node);

    assert_eq!(it.next().unwrap(), 99979);
    assert_eq!(it.next().unwrap(), 99980);
    assert_eq!(it.next().unwrap(), 99981);
    assert_eq!(it.next().unwrap(), 99982);
    assert_eq!(it.next().unwrap(), 99983);
    assert_eq!(it.next().unwrap(), 99984);
    assert_eq!(it.next().unwrap(), 99985);
    assert_eq!(it.next().unwrap(), 99986);
    assert_eq!(it.next().unwrap(), 99987);
    assert_eq!(it.next().unwrap(), 99988);
    assert_eq!(it.next().unwrap(), 99989);
    assert_eq!(it.next().unwrap(), 99990);
    assert_eq!(it.next().unwrap(), 99998);
    assert_eq!(it.next().unwrap(), 99999);
    assert!(it.next().is_none());
}