use super::*;

#[test]
fn test_graph_building() {
    let complete_builder = ImmutableGraphBuilder::<u32>::new()
                            .load_graph("clear.graph.plain")
                            .load_offsets("clear.offsets.plain")
                            .count_nodes()
                            .count_arcs();
    let uncompressed_graph = complete_builder.construct();
    assert_eq!(uncompressed_graph.num_nodes(), 100000);
}