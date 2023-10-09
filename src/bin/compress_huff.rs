use std::{time::Instant, fs::File, io::BufReader};

use clap::Parser;
use webgraph_rust::{properties::Properties, webgraph::bvgraph_huffman_out::BVGraphBuilder, utils::encodings::{GammaCode, UnaryCode, ZetaCode, Huff}, EncodingType, ImmutableGraph};

#[derive(Parser, Debug)]
#[command(about = "Generate a graph having the blocks, the intervals and the residuals Huffman-encoded")]
struct Args {
    /// The basename of the graph file
    source_name: String,
    /// The destination basename of the graph file
    dest_name: String,
}

fn main() {
    let args = Args::parse();

    let properties_file = File::open(format!("{}.properties", args.source_name));
    let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
    let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));
    let props = Properties::from(p);
    
    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.residual_coding) {
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::ZETA) => {},
        _ => panic!("Only the default encoding types sequence (GAMMA, GAMMA, GAMMA, GAMMA, UNARY, ZETA) is supported for Huffman compression")
    };

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
        .load_graph(&args.source_name)
        .load_offsets(&args.source_name)
        .load_outdegrees()
        .build();

    let comp_time = Instant::now();
    bvgraph.store(&args.dest_name).expect("Failed storing the graph");
    let comp_time = comp_time.elapsed().as_nanos() as f64;
    println!("compressed the graph in {}ns", comp_time);
}