use std::time::Instant;

use webgraph_rust::{properties::Properties, webgraph::bvgraph_huffman_out::BVGraphBuilder, utils::{encodings::{GammaCode, UnaryCode, ZetaCode}, huffman::Huff}, ImmutableGraph};

fn main() {
    let source = "../../../eu-2015";
    let dest = "../../../huff_eu";
    let properties_file = std::fs::File::open(format!("{}.properties", source));
    let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", source));
    let p = java_properties::read(std::io::BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));

    let mut props = Properties::from(p);

    assert!(props.nodes as u64 <= u64::MAX, "This version of WebGraph cannot handle graphs with {} (>=2^63) nodes", props.nodes);

    props.window_size = 7;
    props.max_ref_count = 3;
    props.min_interval_len = 4;

    let mut bvgraph = BVGraphBuilder::<
        GammaCode,
        GammaCode,
        GammaCode,
        GammaCode,
        UnaryCode,
        ZetaCode,
        Huff,
        GammaCode,
        GammaCode,
        GammaCode,
        UnaryCode,
        Huff,
    >::new()
        .set_min_interval_len(props.min_interval_len)
        .set_max_ref_count(props.max_ref_count)
        .set_window_size(props.window_size)
        .set_zeta(props.zeta_k)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(source)
        .load_offsets(source)
        .load_outdegrees()
        .build();

    let comp_time = Instant::now();
    bvgraph.store(dest).expect("Failed storing the graph");
    let comp_time = comp_time.elapsed().as_nanos() as f64;
    println!("compressed the graph in {}ns", comp_time);

    // TODO: divide BVGRAPH into reader and writer
    // Update properties
    // Test writing and reading huff
}