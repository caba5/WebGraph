use super::*;

fn build_graph<'a, T: num_traits::PrimInt>() -> UncompressedGraph<T> 
where 
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: DeserializeOwned,
    T: Serialize
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
    T: DeserializeOwned,
    T: Serialize
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
    let mut uncompressed_graph = build_graph_bin::<u32>();
    let it = uncompressed_graph.successors(0);

    let correct = vec![1, 4, 5, 6, 7, 8, 13, 38, 48, 154].into_iter();

    assert!(it.eq(correct));
}

#[test]
fn test_iterate_on_mid_successors() {
    let mut uncompressed_graph = build_graph_bin::<u32>();
    let it = uncompressed_graph.successors(6);

    let correct = vec![0, 1, 4, 5, 6, 7, 8, 13, 38, 84821].into_iter();

    assert!(it.eq(correct));
}

#[test]
fn test_iterate_on_end_successors() {
    let mut uncompressed_graph = build_graph_bin::<u32>();
    let last_node = (uncompressed_graph.n - 1) as u32;  // n-1 since it is the last node
    let it = uncompressed_graph.successors(last_node);

    let correct = 
        vec![99979, 99980, 99981, 99982, 99983, 99984, 99985, 99986, 99987, 99988, 99989, 99990, 99998, 99999].into_iter();

    assert!(it.eq(correct));
}