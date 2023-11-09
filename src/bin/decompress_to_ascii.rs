use std::{fs::File, io::{BufReader, BufWriter, Write}};

use clap::Parser;
use webgraph_rust::{properties::Properties, webgraph::bvgraph::BVGraphBuilder, utils::{encodings::{GammaCode, UnaryCode, ZetaCode}, EncodingType}, ImmutableGraph};

#[derive(Parser, Debug)]
#[command(about = "Generates the ASCII version of the graph (<node_id, posting_list>)")]
struct Args {
    /// The basename of the graph file
    source_name: String,
    /// The basename of the ascii file to be written
    dest_name: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let properties_file = File::open(format!("{}.properties", args.source_name));
    let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
    let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));
    let props = Properties::from(p);
    
    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.interval_coding, props.residual_coding) {
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::GAMMA, EncodingType::ZETA) => {},
        _ => panic!("Only the default encoding types sequence (GAMMA, GAMMA, GAMMA, GAMMA, UNARY, GAMMA, ZETA) is supported for generating offsets")
    };

    let loaded_g = BVGraphBuilder::<
        GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode,
        GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode // Dummy values
    >::new()
        .set_min_interval_len(props.min_interval_len)
        .set_max_ref_count(props.max_ref_count)
        .set_window_size(props.window_size)
        .set_zeta(props.zeta_k)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(&args.source_name)
        .load_outdegrees()
        .build();

    let mut iter = loaded_g.iter();
    let mut n = loaded_g.num_nodes();

    let f = File::create(format!("{}.txt", args.dest_name))?;
    let mut writer = BufWriter::new(f); 

    while n > 0 {
        let node = iter.next().unwrap();
        let outd = iter.outdegree();
        let succ = iter.successor_array();

        writeln!(writer, "{}\t{}", node, succ.iter().take(outd).map(|x| x.to_string()).collect::<Vec<_>>().join("\t"))?;

        n -= 1;
    }
    
    writer.flush()?;

    Ok(())
}