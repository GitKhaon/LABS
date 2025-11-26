use std::time::Instant;
use rayon::prelude::*;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

fn log_to_file(length: usize, time_taken: f64, nodes_visited: usize) -> std::io::Result<()> {
    // Open the file in append mode (or create it if it doesn't exist)
    let mut file = OpenOptions::new()
        .create(true)   // Create the file if it doesn't exist
        .append(true)   // Append to the file instead of overwriting
        .open("skew_brnbo_parallel.txt")?;
    
    // Write the data to the file
    writeln!(
        file, 
        "N: {}, Time: {:.2} seconds, Nodes: {}",
        length, time_taken, nodes_visited
    )?;
    
    Ok(())
}

fn modular_min_abs(a: isize, b: isize) -> isize {
    let mut result = a % b;
    if result > b / 2 {
        result -= b;
    } else if result < -b / 2 {
        result += b;
    }
    result.abs()
}

#[derive(Clone, Debug)]
struct BitSequence {
    sequence: Vec<bool>,
    fixed_energys: Vec<isize>,
    u_min: Vec<isize>,
    u_max: Vec<isize>,
    g: Vec<usize>,
    m: usize,
    n: usize,
}

impl BitSequence {
    fn new(length: usize) -> Self {
        BitSequence {
            sequence: Vec::new(),
            fixed_energys: Vec::new(),
            u_min: Vec::new(),
            u_max: Vec::new(),
            g: Vec::new(),
            m: 0,
            n: length,
        }
    }

    fn initialize_fixed_energys(&mut self) {
        self.fixed_energys = Vec::with_capacity(self.n - 1);
        for _ in 0..self.n - 1 {
            self.fixed_energys.push(0);
        }
    }

    fn multi_update_fixed_energys(&mut self) {
        let m = self.m;
        let n = self.n;

        for k in (1..(n - 1)).step_by(2) {
            for i in 0..m {   //first half
                if i + (k+1) < m {  //if correlate is still in first half
                    if self.sequence[i] ^ self.sequence[i + (k+1)] {
                        self.fixed_energys[k] -= 1;
                    } else {
                        self.fixed_energys[k] += 1;
                    }
                }
                if i + (k+1) >= n-m && i+(k+1) < n {   //if correlate in second half
                    if self.sequence[i] ^ self.sequence[i + (k+1) - (n-2*m)] {
                        self.fixed_energys[k] -= 1;
                    } else {
                        self.fixed_energys[k] += 1;
                    }
                }
                if i + (k+1) == n/2 {
                    if self.sequence[i] {
                        self.fixed_energys[k] += 2;
                    } else {
                        self.fixed_energys[k] -= 2;
                    }
                }
            }
            for i in 0..m { //second half
                if n >= n - m + 1 + i + (k+1) { //if correlate in second half
                    if self.sequence[2*m - 1 - i] ^ self.sequence[2*m - 1 - i - (k+1)] {
                        self.fixed_energys[k] -= 1;
                    } else {
                        self.fixed_energys[k] += 1;
                    }
                }
            }
        }       
    }

    fn multi_update_u_min_max(&mut self) {
        self.u_max = Vec::with_capacity(self.n - 1);
        self.u_min = Vec::with_capacity(self.n - 1);
        for _ in 0..self.n - 1 {
            self.u_max.push(0);
            self.u_min.push(0);
        }
        
        let m = self.m;
        let n = self.n;

        for k in (1..(n - m)).step_by(2) {
            let base_amount = if (k+1) > n/2 { n -(k+1) } else { k+1 }; //how many chains when 0 spins set
            let disappeared = if m > (n-(k+1))/2 { 2*m - (n-(k+1)) } else {0}; //how many chains disappered (fixed)
            let amount = base_amount - disappeared;

            for i in 0..amount {
                let first_chain = if m <= k+1 {0} else {m-(k+1)}; //first chain start
                let chain = if i + first_chain + (k+1) >= n-m { i + first_chain + disappeared } else { i + first_chain }; //current chain start
                let chain_length = if m <= k+1 {((n - (k+1) -1 -chain ) / (k+1)) + 1} else {( (n - m -1 -chain)/(k+1) ) + 1}; 
                let l = (n/2  - chain) % (k+1);

                if l == 0 { //update trough middle
                    if chain >= m || chain + chain_length*(k+1) < n - m {     //if free
                        self.u_max[k] += chain_length as isize;
                        self.u_min[k] += chain_length as isize;
                    } else if chain_length == 2 {  //disappears at first update
                        self.u_max[k] += 0;
                        self.u_min[k] += 0;
                    } else if (chain_length / 2) % 2 == 0 {    //if half length even
                        if self.sequence[first_chain +i] {
                            self.u_max[k] += chain_length as isize;
                            self.u_min[k] += chain_length as isize;
                        } else {
                            self.u_max[k] += chain_length as isize -4;
                            self.u_min[k] += chain_length as isize -4;
                        }
                    } else if self.sequence[first_chain +i] {   //if half length odd
                        self.u_max[k] += chain_length as isize;
                        self.u_min[k] += chain_length as isize -4;
                    } else {
                        self.u_max[k] += chain_length as isize -4;
                        self.u_min[k] += chain_length as isize;
                    }

                } else if l == (k+1) / 2 {    //if trough n/2+1 - k/2
                    if chain >= m || chain + chain_length*(k+1) < n - m {          //is free
                        if l % 2 == 0 {
                            self.u_max[k] += chain_length as isize;
                            self.u_min[k] += chain_length as isize -2;  //even if free, this chains back depends on front
                        } else {
                            self.u_max[k] += chain_length as isize -2;
                            self.u_min[k] += chain_length as isize;
                        }
                    } else if m - 1 + (k+1) > n/2 +1 {   //chain disappears
                        self.g[k] = 8;
                        self.u_max[k] += 0;
                        self.u_min[k] -= 0;
                    } else if l % 2 == 0 {   //if k/2 /2 even back is same as front
                        self.u_max[k] += chain_length as isize;
                        self.u_min[k] += chain_length as isize -2;
                        self.g[k] = 4; 
                    } else {     //otherwise complement of front
                        self.u_max[k] += chain_length as isize -2;
                        self.u_min[k] += chain_length as isize;
                        self.g[k] = 4;
                    }
                } else if chain >= m || chain + chain_length*(k+1) < n - m { //free
                    self.u_max[k] += chain_length as isize;
                    self.u_min[k] += chain_length as isize;
                } else if chain_length % 2 == 0 {
                    if self.sequence[first_chain +i] ^ self.sequence[first_chain + i + chain_length*(k+1) - (n-2*m)] {
                        self.u_max[k] += chain_length as isize -2;
                        self.u_min[k] += chain_length as isize -2;
                    } else {
                        self.u_max[k] += chain_length as isize;
                        self.u_min[k] += chain_length as isize;
                    }
                } else if self.sequence[first_chain +i] ^ self.sequence[first_chain + i + chain_length*(k+1) - (n-2*m)] {
                    self.u_max[k] += chain_length as isize -2;
                    self.u_min[k] += chain_length as isize;
                } else {
                    self.u_max[k] += chain_length as isize;
                    self.u_min[k] += chain_length as isize -2;
                }
            }
        }
    }

    fn initialize_g(&mut self) {
        self.g = Vec::with_capacity(self.n - 1);
        for _ in 0..self.n - 1 {
            self.g.push(2);
        }
    }

    fn set_bits(&mut self, left: bool, right: bool) {
        let mid = self.sequence.len() / 2;
        self.sequence.insert(mid, left);
        self.sequence.insert(mid + 1, right);
        self.m += 1;
    }

    fn update_fixed_energys(&mut self) {
        let m = self.m;
        let n = self.n;

        for k in (1..(n - m)).step_by(2) {
        
            if m + k >= n - m {
                if self.sequence[m - 1] == self.sequence[3 * m + k - n] {
                    self.fixed_energys[k] += 1;
                } else {
                    self.fixed_energys[k] -= 1;
                }
            }

            if m + k > n - m {
                if self.sequence[m] == self.sequence[n - m - (k + 1)] {
                    self.fixed_energys[k] += 1;
                } else {
                    self.fixed_energys[k] -= 1;
                }
            }

            if m >= k + 2 {
                if self.sequence[m - 1] == self.sequence[m - (k + 2)] {
                    self.fixed_energys[k] += 2;
                } else {
                    self.fixed_energys[k] -= 2;
                }
            }

            if m+k == n/2 {
                if self.sequence[m-1] {
                    self.fixed_energys[k] += 2;
                } else {
                    self.fixed_energys[k] -= 2;
                }
            }
        }
    }

    fn update_u_min_max(&mut self) {

        let m = self.m;
        let n = self.n;

        self.u_max[n - m - 1] = 0;
        self.u_min[n - m - 1] = 0;

        for k in (1..(n - m - 1)).step_by(2) {       //update doubles 
    
            let l = ((n/2 +1) - m) % (k+1);

            if l == 0 {                  //case if through middle
                if m <= (k+1){           //the first k updates are the first update for each chain. This chain goes from free to fixed.
                                         //if m+k is n/2 +1 the first update makes the chain disappear
                    if m + (k+1) == n/2 + 1 {
                        self.u_max[k] -= 2;
                        self.u_min[k] -= 2;
                        continue;
                    }
                    if self.sequence[m-1] {
                        if (((n/2 -m)/ (k+1)) + 1) % 2 == 0{   //since middle is one, need to check, dependent on if first half is even or odd
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 0;
                            continue;
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 4;
                            continue;
                        }
                    } else if (((n/2 -m)/ (k+1)) + 1) % 2 == 0{
                        self.u_max[k] -= 4;
                        self.u_min[k] -= 4;
                        continue;
                    } else {
                        self.u_max[k] -= 4;
                        self.u_min[k] -= 0;
                        continue;
                    }
                } else {
                    if n/2 -m < (k+1) {
                        if self.sequence[m-1 -(k+1)] {
                            self.u_max[k] -= 4;
                            self.u_min[k] -= 4;
                            continue;
                        } else{
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 0;
                            continue;
                        }
                    }
                    if self.sequence[m-1-(k+1)] {    //from +..+
                        if self.sequence[m-1] {      //+..+ to +..+
                            if (((n/2 -m)/ (k+1)) + 1) % 2 == 0{  //check if half length odd or even
                                self.u_max[k] -= 2;
                                self.u_min[k] += 2;
                                continue;
                            } else {
                                self.u_max[k] -= 2;
                                self.u_min[k] -= 6;
                                continue;
    
                            }
                        } else {      //+..+ to -..-
                            self.u_max[k] -= 6;
                            self.u_min[k] -= 2;
                            continue;
                        }
                    } else {         //from -..-
                        if self.sequence[m-1] {   //-..- to +..+
                            self.u_max[k] += 2;
                            self.u_min[k] -= 2;
                            continue;
                        } else {    //-..- to -..-
                            if (((n/2 -m)/ (k+1)) + 1) % 2 == 0{
                                self.u_max[k] -= 2;
                                self.u_min[k] -= 6;
                                continue;
                            } else {
                                self.u_max[k] -= 2;
                                self.u_min[k] += 2;
                                continue;
    
                            }
                        }
                    }
                }
            } else if l == (k+1) / 2 {       //case if through k/2
                if m - 1 + (k+1) > n/2 +1{   //in this case s_a = n/2 -l and s_b = n/2 + l. The chain dissapears.
                    self.g[k] = 8;
                    if l % 2 == 0 {
                        if m <= (k+1){
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1;
                            continue;
                        } else {
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1;
                            continue;
                        }
                    } else if m <= (k+1){
                        self.u_max[k] += 1;
                        self.u_min[k] -= 1;
                        continue;
                    } else {
                        self.u_max[k] -= 1;
                        self.u_min[k] -= 3;
                        continue;
                    }
                } else if m <= (k+1){
                    self.g[k] = 4;
                    self.u_max[k] -= 0;
                    self.u_min[k] -= 0;
                    continue;
                } else if n - 2*m < (k+1) {
                    if l % 2 == 0 {
                        self.u_max[k] -= 3;
                        self.u_min[k] -= 1;
                        continue;
                    } else {
                        self.u_max[k] -= 1;
                        self.u_min[k] -= 3;
                        continue;
                    }
                } else {
                    self.u_max[k] -= 2;
                    self.u_min[k] -= 2;
                    continue;
                }
            } else if m  <= (k+1) {       // m <= k check if free -> fixed
                let chain_length = ((n - (k+1) - 1 -(m-1)) / (k+1)) + 1;
                
                if m + chain_length*(k+1) > n - (m-1) {
                    if chain_length == 1 {
                        self.u_max[k] -= 2;
                        self.u_min[k] -= 2;
                        continue;
                    } else if self.sequence[m-1] == self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] { //after update
                        if chain_length % 2 == 0 {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 0;
                            continue; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 4;
                            continue;
                        }

                    } else if chain_length % 2 == 0 {
                        self.u_max[k] -= 4;
                        self.u_min[k] -= 4; 
                        continue;
                    } else {
                        self.u_max[k] -= 4;
                        self.u_min[k] -= 0;
                        continue;                        
                    }
                } else {
                    self.u_max[k] -= 0;
                    self.u_min[k] -= 0;
                    continue;
                } 

            } else {                                                                                    //m > k Update chains.
                let chain_length = ( (n - 2*m + 1)/(k+1) ) + 1;

                if chain_length == 1 {
                    if self.sequence[m - 1 - (k+1)] == self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                        self.u_max[k] -= 4;
                        self.u_min[k] -= 4;
                        continue;
                    } else {
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 0;
                        continue;
                    }
                } else if self.sequence[m - 1 - (k+1)] == self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {     //if first update was equal 
                    if self.sequence[m - 1] == self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {         //if first stays equal
                        if chain_length % 2 == 0{                                                      //if chain length is even
                            self.u_max[k] -= 2;
                            self.u_min[k] += 2; 
                            continue;
                        } else {                                                                       //if chain length is odd
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 6;
                            continue;
                        }
                    } else {                                                                           //if first goes to unequal
                        if chain_length % 2 == 0{                                                      //if chain length is even
                            self.u_max[k] -= 6;
                            self.u_min[k] -= 2; 
                            continue;
                        } else {                                                                       //if chain length is odd
                            self.u_max[k] -= 6;
                            self.u_min[k] -= 2;
                            continue;
                        }
                    }
                } else {                                                                               //if first update was unequal
                    if self.sequence[m - 1] == self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {         //if first goes to equal
                        if chain_length % 2 == 0{                                                      //if chain length is even
                            self.u_max[k] += 2;
                            self.u_min[k] -= 2; 
                            continue;
                        } else {                                                                       //if chain length is odd
                            self.u_max[k] += 2;
                            self.u_min[k] -= 2;
                            continue;
                        }
                    } else {                                                                           //if first stays unequal
                        if chain_length % 2 == 0{                                                      //if chain length is even
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 6; 
                            continue;
                        } else {                                                                       //if chain length is odd
                            self.u_max[k] -= 2;
                            self.u_min[k] += 2;
                            continue;
                        }
                    }
                }
            }  
        }
    }

    fn set_last_bit(&mut self, bit: bool) {
        let mid = self.sequence.len() / 2;
        self.sequence.insert(mid, bit);
        self.m += 1;
    }
}


fn branch(individual: &mut BitSequence, bound: isize) -> Vec<BitSequence> {
    let mut children = Vec::new();

    if (individual.n/2 - individual.m) % 2 == 0 {
        let mut child_tt = individual.clone();
        child_tt.set_bits(true, true);
        child_tt.update_fixed_energys();
        child_tt.update_u_min_max();
        let e_min_tt = calculate_e_min(&mut child_tt);
        if e_min_tt <= bound {
            children.push((e_min_tt, child_tt));
        }

        let mut child_ff = individual.clone();
        child_ff.set_bits(false, false);
        child_ff.update_fixed_energys();
        child_ff.update_u_min_max();
        let e_min_ff = calculate_e_min(&mut child_ff);
        if e_min_ff <= bound {
            children.push((e_min_ff, child_ff));
        }

    } else {
        let mut child_tf = individual.clone();
        child_tf.set_bits(true, false);
        child_tf.update_fixed_energys();
        child_tf.update_u_min_max();
        let e_min_tf = calculate_e_min(&mut child_tf);
        if e_min_tf <= bound {
            children.push((e_min_tf, child_tf));
        }

        let mut child_ft = individual.clone();
        child_ft.set_bits(false, true);
        child_ft.update_fixed_energys();
        child_ft.update_u_min_max();
        let e_min_ft = calculate_e_min(&mut child_ft);
        if e_min_ft <= bound {
            children.push((e_min_ft, child_ft));
        }
    }

    children.sort_by_key(|&(e_min, _)| -e_min);
    children.into_iter().map(|(_, child)| child).collect()
}

fn calculate_e_min(individual: &mut BitSequence) -> isize {
    let mut result = vec![0; individual.n - 1];

    for (k, item) in result.iter_mut().enumerate().take(individual.n - 1) {
        if k % 2 == 0 {
            *item = 0;
            continue;
        }
        if individual.fixed_energys[k] <= -(individual.u_max[k]) {
            *item = individual.fixed_energys[k] + individual.u_max[k];
            
        } else if individual.fixed_energys[k] >= individual.u_min[k] {
            *item = individual.fixed_energys[k] - individual.u_min[k];
            
        } else {
            *item = modular_min_abs(individual.fixed_energys[k] - individual.u_min[k], individual.g[k] as isize);
        }
    }
    let energy: isize = result.iter().map(|&x| x * x).sum();
    energy
}

fn final_energy(individual: &BitSequence) -> isize {
    let mut result = 0;
    for k in 0..(individual.n - 1) {
        result += individual.fixed_energys[k] * individual.fixed_energys[k];
    }
    result
}

fn reverse(individual: &mut BitSequence) {
    individual.sequence.reverse();
}

fn complement(individual: &mut BitSequence) {
    for bit in &mut individual.sequence {
        *bit = !*bit;
    }
}

fn alternate_complement(individual: &mut BitSequence) {
    let mid = individual.sequence.len() / 2;
    for i in 0..mid {
        if i % 2 == 0 {
            individual.sequence[i] = !individual.sequence[i];
        }
    }
    for i in 0..mid {
        if i % 2 != individual.n % 2 {
            individual.sequence[2*mid -1 -i] = !individual.sequence[2*mid -1 -i];
        }
    }
}

fn main() {

    let args: Vec<String> = env::args().collect();

    let length: usize = args.get(1)
        .expect("Provide N")
        .parse()
        .expect("N as usize");
    let bound: isize = args.get(2)
        .expect("Provide bound")
        .parse()
        .expect("bound as isize");
    
    let bound = AtomicIsize::new(bound);
    
    let mut roots = Vec::new();
    
    let two_m = 20;

    let time_roots = Instant::now();
    
    for i in 0..(1 << (two_m/2)) {
        let mut sequence = Vec::new();
        for j in 0..two_m/2 {
            sequence.push((i & (1 << j)) != 0); // First n/2 spins
        }
        for j in 0..two_m/2 {
            if (length/2) % 2 == 0 {
                if j % 2 == 0 {
                    sequence.insert(two_m/2, sequence[j]);
                } else {
                    sequence.insert(two_m/2, !sequence[j]);
                }
            } else if j % 2 == 0 {
                sequence.insert(two_m/2, !sequence[j]);
            } else {
                sequence.insert(two_m/2, sequence[j]);
            }
        }


        let mut individual = BitSequence::new(length);
        individual.m = two_m/2;
        individual.sequence = sequence;

        if individual.n % 2 == 1 {
            let mut r = individual.clone();
            reverse(&mut r);
            let mut c = individual.clone();
            complement(&mut c);
            let mut a = individual.clone();
            alternate_complement(&mut a);
            let mut rc = r.clone();
            complement(&mut rc);
            let mut ra = r.clone();
            alternate_complement(&mut ra);
            let mut ca = c.clone();
            alternate_complement(&mut ca);
            let mut rca = rc.clone();
            alternate_complement(&mut rca);

            let mut all_sequences = [
                individual.sequence.clone(),
                r.sequence.clone(),
                c.sequence.clone(),
                a.sequence.clone(),
                rc.sequence.clone(),
                ra.sequence.clone(),
                ca.sequence.clone(),
                rca.sequence.clone(),
            ];
        
            all_sequences.sort();
            let smallest_sequence = all_sequences.first().unwrap();

            if smallest_sequence == &individual.sequence {
                individual.initialize_fixed_energys();
                individual.initialize_g();
                individual.multi_update_fixed_energys();
                individual.multi_update_u_min_max();
                roots.push(individual);
                c.initialize_fixed_energys();
                c.initialize_g();
                c.multi_update_fixed_energys();
                c.multi_update_u_min_max();
                roots.push(c);

            }
        }
    }

    let duration_roots = time_roots.elapsed();
    println!("{} Roots created in {} seconds", roots.len(), duration_roots.as_secs());

    let total_nodes = AtomicUsize::new(0);

    let time_total = Instant::now();

    roots.par_iter_mut().for_each(|root| {
        
        let mut individuals = vec![root.clone()];

        let mut nodes_visited = 0;

        while let Some(mut individual) = individuals.pop() {

            if individual.m == individual.n / 2 {
                let mut child_t = individual.clone();
                child_t.set_last_bit(true);
                total_nodes.fetch_add(nodes_visited, Ordering::Relaxed);
                let e_min = final_energy(&child_t);
                if e_min <= bound.load(Ordering::Relaxed) {
                    let duration = time_total.elapsed().as_secs_f64();
                    println!("Sequence found in {} seconds {} nodes, with energy {}", duration, nodes_visited, e_min);
                    let _ = log_to_file(length, duration, nodes_visited);
                    bound.store(e_min, Ordering::Relaxed);
                }

            } else {
                let children = branch(&mut individual, bound.load(Ordering::Relaxed));
                nodes_visited += 2;
                individuals.extend(children);
            }
        }
        total_nodes.fetch_add(nodes_visited, Ordering::Relaxed);
    });
    let total_duration = time_total.elapsed().as_secs_f64();
    println!("Tree explored in {} and {} nodes", total_duration, total_nodes.load(Ordering::Relaxed));
    let _ = log_to_file(length, total_duration, total_nodes.load(Ordering::Relaxed));
}