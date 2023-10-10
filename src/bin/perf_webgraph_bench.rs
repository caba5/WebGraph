use std::{time::Instant, fs::File, io::BufReader};

use clap::Parser;

use rand::Rng;
use webgraph_rust::{EncodingType, utils::encodings::{UniversalCode, GammaCode, UnaryCode, ZetaCode}, webgraph::bvgraph::{BVGraph, BVGraphBuilder}, ImmutableGraph, properties::Properties};

#[derive(Parser, Debug)]
struct WGArgs {
    /// The size of the window
    #[arg(short = 'w', long = "window-size", default_value_t = 7)]
    window_size: usize,
    /// The maximum reference chain length
    #[arg(short = 'r', long = "max-ref-count", default_value_t = 3)]
    max_ref_count: usize,
    /// The minimum length of the interval
    #[arg(short = 'i', long = "min-interval-len", default_value_t = 4)]
    min_interval_length: usize,
    /// Specifies the block coding type
    #[arg(long = "block-coding", default_value_t = EncodingType::GAMMA)]
    block_coding: EncodingType,
    /// Specifies the block count coding type
    #[arg(long = "block-count-coding", default_value_t = EncodingType::GAMMA)]
    block_count_coding: EncodingType,
    /// Specifies the outdegree coding type
    #[arg(long = "outdegree-coding", default_value_t = EncodingType::GAMMA)]
    outdegree_coding: EncodingType,
    /// Specifies the offset coding type
    #[arg(long = "offset-coding", default_value_t = EncodingType::GAMMA)]
    offset_coding: EncodingType,
    /// Specifies the reference coding type
    #[arg(long = "reference-coding", default_value_t = EncodingType::UNARY)]
    reference_coding: EncodingType,
    /// Specifies the residual coding type
    #[arg(long = "residual-coding", default_value_t = EncodingType::ZETA)]
    residual_coding: EncodingType,
    /// Specifies the k parameter for ZetaK coding
    #[arg(short = 'k', long = "zetak", default_value_t = 3)]
    zeta_k: usize,
    /// Whether to compress offsets through Elias-Fano. It overwrites the offset_coding, if specified.
    #[arg(long = "ef", default_value_t = false)]
    elias_fano: bool,
    /// Source basename
    source_name: String,                                                    // TODO: PathBuf instead of str?
    /// Destination basename
    dest_name: Option<String>,                                              // TODO: PathBuf instead of str?
}

const N_QUERIES: usize = 1000000;

fn gen_queries(n_queries: usize, range_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    (0..n_queries)
        .map(|_| rng.gen_range(0..range_size))
        .collect()
}

fn decompression_perf_test<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOudegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOudegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
>(bvgraph: &mut BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOudegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOudegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutResidualCoding,
>) {
    let n = bvgraph.num_nodes();

    let queries = gen_queries(N_QUERIES, n - 1);

    let total = Instant::now();
    for &query in queries.iter() {
        bvgraph.successors(query);
    }
    let avg_query = (total.elapsed().as_nanos() as f64) / N_QUERIES as f64;

    println!("time per query: {}ns", avg_query);
}

fn create_graph<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOudegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOudegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
>(props: &Properties, in_name: &str, out_name: Option<String>, elias_fano: bool) {
    let mut bvgraph = BVGraphBuilder::<
        InBlockCoding,
        InBlockCountCoding,
        InOudegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOudegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutResidualCoding,
    >::new()
        .set_min_interval_len(props.min_interval_len)
        .set_max_ref_count(props.max_ref_count)
        .set_window_size(props.window_size)
        .set_zeta(props.zeta_k)
        .set_elias_fano(elias_fano)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(in_name)
        .load_offsets(in_name)
        .load_outdegrees()
        .build();

    if let Some(out_name) = out_name{
        let comp_time = Instant::now();
        bvgraph.store(out_name.as_str()).expect("Failed storing the graph");
        let comp_time = comp_time.elapsed().as_nanos() as f64;
        println!("compressed the graph in {}ns", comp_time);
    } else {
        decompression_perf_test(&mut bvgraph);    
    }
}

fn main() {
    let mut args = WGArgs::parse();
    let mut props = Properties::default();
    
    let properties_file = File::open(format!("{}.properties", args.source_name));
    let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
    let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));

    props = Properties::from(p);

    assert!(props.nodes as u64 <= u64::MAX, "This version of WebGraph cannot handle graphs with {} (>=2^63) nodes", props.nodes);

    props.window_size = args.window_size;
    props.max_ref_count = args.max_ref_count;
    props.min_interval_len = args.min_interval_length;
    
    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.residual_coding, 
            args.block_coding, args.block_count_coding, args.outdegree_coding, args.offset_coding, args.reference_coding, args.residual_coding) {
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::ZETA, // Default case
        EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::ZETA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, ZetaCode, GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, ZetaCode>
            (&props, &args.source_name, args.dest_name, args.elias_fano),
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::ZETA, // Default to gamma
        EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA) => 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, ZetaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode, GammaCode>
            (&props, &args.source_name, args.dest_name, args.elias_fano),
        _ => panic!("Unexpected encoding types", )
    }
}