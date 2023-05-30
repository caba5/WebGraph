use super::*;

fn build_graph<'a, T: num_traits::PrimInt>() -> UncompressedGraph<T> 
where 
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: DeserializeOwned
{
    ImmutableGraphBuilder::<T>::new()
                            .load_graph("clear.graph.plain")
                            .load_offsets("clear.offsets.plain")
                            .count_nodes()
                            .count_arcs()
                            .construct()
}

fn build_graph_bin<'a, T: num_traits::PrimInt>() -> UncompressedGraph<T> 
where 
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: DeserializeOwned
{
    ImmutableGraphBuilder::<T>::new()
                            .load_graph_bin("clear.graph.plain")
                            .load_offsets_bin("clear.offsets.plain")
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