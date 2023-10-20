use std::{time::Instant, fs::File, io::BufReader};

use clap::Parser;
use webgraph_rust::{properties::Properties, webgraph::bvgraph_huffman_in::BVGraphBuilder, utils::encodings::{GammaCode, UnaryCode, ZetaCode, Huff}, EncodingType, ImmutableGraph};

#[derive(Parser, Debug)]
#[command(about = "Generate a graph having the blocks, the intervals and the residuals Huffman-encoded")]
struct Args {
    /// The basename of the huffman-compressed graph file
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
    
    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.interval_coding, props.residual_coding) {
        (EncodingType::HUFFMAN, EncodingType::GAMMA, EncodingType::HUFFMAN, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::HUFFMAN, EncodingType::HUFFMAN) => {},
        _ => panic!("The graph has to be Huffman-encoded (that is, its blocks, intervals, and residuals must be encoded by Huffman)")
    };

    let mut bvgraph = BVGraphBuilder::<
        Huff, GammaCode, Huff, GammaCode, UnaryCode, Huff, Huff,
        // Default encoding
        GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode
    >::new()
        .set_min_interval_len(props.min_interval_len)
        .set_max_ref_count(props.max_ref_count)
        .set_window_size(props.window_size)
        .set_zeta(props.zeta_k)
        .set_huff_outdegrees_parameters(props.huff_outdegrees_bits)
        .set_huff_blocks_parameters(props.huff_blocks_bits)
        .set_huff_residuals_parameters(props.huff_residuals_bits)
        .set_huff_intervals_left_parameters(props.huff_intervals_left_bits)
        .set_huff_intervals_len_parameters(props.huff_intervals_len_bits)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(&args.source_name)
        .load_offsets(&args.source_name)
        .load_outdegrees()
        .build();

    let comp_time = Instant::now();
    bvgraph.store(&args.dest_name).expect("Failed storing the graph");
    let comp_time = comp_time.elapsed().as_nanos() as f64;
    println!("decompressed the graph in {}ns", comp_time);
}