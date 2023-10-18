use std::{fs, str::FromStr, fmt::{self, Debug}, marker::PhantomData};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

use crate::ImmutableGraph;

#[derive(Serialize, Deserialize, Clone)]
pub struct AsciiGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Copy
{
    n: usize,
    m: usize,
    pub graph_memory: Vec<T>,
    pub offsets: Vec<usize>,
}

impl<T> ImmutableGraph for AsciiGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Copy
{
    type NodeT = T;

    #[inline(always)]
    fn num_nodes(&self) -> usize {
        self.n
    }

    #[inline(always)]
    fn num_arcs(&self) -> usize {
        self.m
    }

    /// Returns the number of adjacent nodes of the given node (if in bounds).
    /// 
    /// Since the nodes are numbered from *0* to *n - 1*, asking for the outdegree of node *x* 
    /// corresponds to asking the outdegree of the *x*-th node (zero-indexed).
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number (from *0* to *n - 1*)
    /// 
    /// # Examples
    /// 
    /// ```
    /// let ascii_graph = build_graph_bin::<u32>();
    /// ascii_graph.outdegree(5);
    /// ```
    #[inline(always)]
    fn outdegree(&mut self, x: Self::NodeT) -> Option<usize> {
        // Offsets also represent nodes having no out-edges.
        // The argument node has to be lower than the last offsets position since the latter
        // is left just for convenience and does not represent any real position in the graph.
        if x < T::zero() || x.to_usize().unwrap() >= self.offsets.len() {
            return None;
        }
        
        let usized_ind = x.to_usize().unwrap();

        self.graph_memory[self.offsets[usized_ind] + 1].to_usize()
    }

    /// Stores both `graph_memory` and `offsets` into their respective files.
    /// 
    /// # Arguments
    /// 
    /// * `basename` - The name (with or without path) the saved files will have 
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut ascii_graph = AsciiGraphBuilder::new()
    ///     .load_ascii(&("src/ascii_graph/test_data/".to_owned() + "cnr-2000-ascii"))
    ///     .build();
    /// let result = ascii_graph.store("graph_name");
    /// ```
    fn store(&mut self, basename: &str) -> std::io::Result<()>{
        assert_ne!(basename, "");

        fs::write(format!("{}.graph.bin", basename), bincode::serialize(&self.graph_memory).unwrap())?;
        fs::write(format!("{}.offsets.bin", basename), bincode::serialize(&self.offsets).unwrap())?;

        Ok(())
    }
}

impl<T> AsciiGraph<T> 
where T: 
        num_traits::Num
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Copy
{
    /// Returns an `AsciiGraphIterator` over the successors of `x`.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number (from *0* to *n - 1*)
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut ascii_graph = AsciiGraphBuilder::new()
    ///     .load_ascii(&("src/ascii_graph/test_data/".to_owned() + "cnr-2000-ascii"))
    ///     .build();
    /// let successors_it = ascii_graph.successors(5);
    /// 
    /// while let Some(node_val) = successors_it.next() {
    ///     println!("Successor: {}", node_val);
    /// }
    /// ```
    #[inline(always)]
    pub fn successors(&mut self, x: T) -> AsciiGraphIterator<T, &AsciiGraph<T>> {
        assert!(x >= T::zero() && x.to_usize().unwrap() < self.n, "The node has to be in the range [0, {})", self.n);

        let base = self.offsets[x.to_usize().unwrap()];
        AsciiGraphIterator { 
            base,
            idx_from_base: 2, // Skip the outdegree
            up_to: base + self.outdegree(x).unwrap() + 1,
            graph: self,
            _phantom: PhantomData
        }
    }

    #[inline(always)]
    pub fn outdegree_unchecked(&self, x: T) -> usize {
        let usized_ind = x.to_usize().unwrap();

        self.graph_memory[self.offsets[usized_ind] + 1].to_usize().unwrap()
    }

    #[inline(always)]
    pub fn iter(&self) -> AsciiGraphIterator<T, &AsciiGraph<T>>{
        AsciiGraphIterator {
            base: 0,
            idx_from_base: 0,
            up_to: self.graph_memory.len() - 1,
            graph: self,
            _phantom: PhantomData
        }
    }
}

impl<T> AsRef<AsciiGraph<T>> for AsciiGraph<T>
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Copy
{
    fn as_ref(&self) -> &AsciiGraph<T> {
        self
    }
}

impl<T> IntoIterator for AsciiGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Copy
{
    type Item = T;

    type IntoIter = AsciiGraphIterator<T, AsciiGraph<T>>;

    fn into_iter(self) -> Self::IntoIter {
        AsciiGraphIterator {
            base: 0,
            idx_from_base: 0,
            up_to: self.graph_memory.len() - 1,
            graph: self,
            _phantom: PhantomData
        }
    }
}

pub struct AsciiGraphIterator<T, UG: AsRef<AsciiGraph<T>>> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Copy
{
    base: usize,
    idx_from_base: usize,
    up_to: usize,
    graph: UG,
    _phantom: PhantomData<T>
}

impl<T, UG: AsRef<AsciiGraph<T>>> Iterator for AsciiGraphIterator<T, UG>
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive 
        + serde::Serialize
        + Copy
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next() {
            return None;
        }

        let g = self.graph.as_ref();
        
        let res = g.graph_memory[self.base + self.idx_from_base];
        
        self.idx_from_base += 1;

        Some(res)
    }
}

impl<T, UG: AsRef<AsciiGraph<T>>> AsciiGraphIterator<T, UG>
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive 
        + serde::Serialize
        + Copy
{
    #[inline(always)]
    pub fn has_next(&self) -> bool {
        self.base + self.idx_from_base <= self.up_to
    }
}

#[derive(Serialize, Deserialize)]
pub struct AsciiGraphBuilder<T> {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Vec<T>,
    loaded_offsets: Vec<usize>,
}

impl<T> Default for AsciiGraphBuilder<T>
where
    T: num_traits::PrimInt + 
    FromStr +
    Serialize +
    DeserializeOwned +
    Debug,
    <T as FromStr>::Err: fmt::Debug,
{
    fn default() -> Self {
        Self { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Vec::default(), 
            loaded_offsets: Vec::default() 
        }
    }
}

impl<T> AsciiGraphBuilder<T>
where 
    T: num_traits::PrimInt + 
        FromStr +
        Serialize +
        DeserializeOwned +
        Debug,
    <T as FromStr>::Err: fmt::Debug,
{
    pub fn new() -> AsciiGraphBuilder<T> {
        Self::default()
    }

    /// Loads an ASCII file containing, for each line, the node followed by the list of its successors.
    /// 
    /// # Arguments
    /// 
    /// * `basename` - The basename of the ASCII graph
    pub fn load_ascii(mut self, basename: &str) -> Self {
        let data = fs::read_to_string(basename)
                            .expect("Failed to load the graph file");
        let mut lines: Vec<&str> = data.split('\n').collect();
        if lines.last().unwrap().is_empty() {
            lines.pop();
        }
        
        let num_nodes = lines.len();
        let mut edges = 0;
        let mut nodes = Vec::with_capacity(num_nodes);
        let mut offsets = vec![0; num_nodes]; // The last offset is just a limit

        // Temp vector which will contain the successors of a node until its outdegree isn't inserted
        let mut temp_succs = Vec::new();

        let mut nodes_idx = 0;

        for line in lines {
            let mut node_list = line.split('\t');

            // Since the file also represents nodes without connections, 
            // we are sure that at least one node exists in the parse
            let node: T = node_list.next().unwrap().parse().unwrap();

            offsets[node.to_usize().unwrap()] = nodes_idx;

            nodes.push(node);
            nodes_idx += 1;
            
            for succ in node_list {
                if !succ.is_empty() {
                    let succ = succ.parse().unwrap();
                    edges += 1;
                    temp_succs.push(succ);
                    nodes_idx += 1;
                }
            }

            nodes.push(T::from(temp_succs.len()).unwrap()); // Push the outdegree
            nodes_idx += 1;
            nodes.append(&mut temp_succs); // Push the successors
        }

        self.num_nodes = num_nodes;
        self.num_edges = edges;
        self.loaded_graph = nodes;
        self.loaded_offsets = offsets;

        self
    }

    /// Loads a serialized graph file.
    /// 
    /// # Arguments
    /// 
    /// * `basename` - The basename of the graph file
    pub fn load_graph_bin(mut self, basename: &str) -> Self {
        let file = fs::read(format!("{}.graph.bin", basename)).expect("Failed reading the graph file");
        
        self.loaded_graph = bincode::deserialize(&file).expect("Error in deserializing the graph file");

        self
    }

    /// Loads a serialized offsets file.
    /// 
    /// # Arguments
    /// 
    /// * `basename` - The basename of the offsets file
    pub fn load_offsets_bin(mut self, basename: &str) -> Self {
        let f = fs::read(format!("{}.offsets.bin", basename)).expect("Failed reading the offsets file");
        self.loaded_offsets = bincode::deserialize(&f).expect("Error in deserializing the offsets file");

        self
    }

    /// Computes the number of nodes by counting the offsets' file entries.
    /// 
    /// This is correct since all the nodes are represented in the graph file, even those
    /// not having any successor.
    pub fn count_nodes(mut self) -> Self {
        assert!(!self.loaded_offsets.is_empty(), "The offsets have to be loaded");

        self.num_nodes = self.loaded_offsets.len();
        self
    }

    /// Computes the number of edges of the graph.
    pub fn count_edges(mut self) -> Self {
        assert!(!self.loaded_graph.is_empty(), "The graph has to be loaded");
        assert!(!self.loaded_offsets.is_empty(), "The offsets have to be loaded");

        // Total number of entries - nodes entries - outdegree entries
        self.num_edges = self.loaded_graph.len() - 2 * self.loaded_offsets.len();

        self
    }

    /// Constructs the AsciiGraph object.
    pub fn build(self) -> AsciiGraph<T> {
        AsciiGraph::<T> { 
            n: self.num_nodes, 
            m: self.num_edges, 
            graph_memory: self.loaded_graph, 
            offsets: self.loaded_offsets 
        }
    }
}

#[cfg(test)]
mod tests;