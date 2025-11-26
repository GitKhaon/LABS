use std::time::Instant;
use std::collections::HashSet;
use rayon::prelude::*;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

fn log_to_file(length: usize, time_taken: f64, nodes_visited: usize, calls_set_min_max: usize) -> std::io::Result<()> {
    // Open the file in append mode (or create it if it doesn't exist)
    let mut file = OpenOptions::new()
        .create(true)   // Create the file if it doesn't exist
        .append(true)   // Append to the file instead of overwriting
        .open("normal_brnbo_symmetry_fixedchains.txt")?;
    
    // Write the data to the file
    writeln!(
        file, 
        "N: {}, Time: {:.2} seconds, Nodes: {}, Calls to set min max {}",
        length, time_taken, nodes_visited, calls_set_min_max
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

    u_min: Vec<usize>,
    u_max: Vec<usize>,
    free_spins_plus: HashSet<usize>,
    free_spins_minus: HashSet<usize>,
    m: usize,
    n: usize,
    branched_pp: bool,
    branched_pm: bool,
    branched_mp: bool,
    branched_mm: bool,
}

impl BitSequence {
    fn new(length: usize) -> Self {
        BitSequence {
            sequence: Vec::new(),
            fixed_energys: Vec::new(),
            u_min: Vec::new(),
            u_max: Vec::new(),
            free_spins_plus: HashSet::new(),
            free_spins_minus: HashSet::new(),
            m: 0,
            n: length,
            branched_pp: false,
            branched_pm: false,
            branched_mp: false,
            branched_mm: false,
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

        for k in 0..(n-1) {
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

        for k in 0..n-m-1 {
            let base_amount = if (k+1) > n/2 { n -(k+1) } else { k+1 }; 
            let disappeared = if m > (n-(k+1))/2 { 2*m - (n-(k+1)) } else {0};
            let amount = base_amount - disappeared;

            for i in 0..amount {
                let first_chain = if m <= k+1 {0} else {m-(k+1)};
                let chain = if i + first_chain + (k+1) >= n-m { i + first_chain + disappeared } else { i + first_chain };
                let chain_length = if m <= k+1 {((n - (k+1) -1 -chain ) / (k+1)) + 1} else {( (n - m -1 -chain)/(k+1) ) + 1}; 

                if chain >= m || chain + chain_length*(k+1) < n - m {
                    self.u_max[k] += chain_length;
                    self.u_min[k] += chain_length;
                } else if chain_length % 2 == 0 {
                    if self.sequence[first_chain +i] ^ self.sequence[first_chain + i + chain_length*(k+1) - (n-2*m)] {
                        self.u_max[k] += chain_length -2;
                        self.u_min[k] += chain_length -2;
                    } else {
                        self.u_max[k] += chain_length;
                        self.u_min[k] += chain_length;
                    }
                } else if self.sequence[first_chain +i] ^ self.sequence[first_chain + i + chain_length*(k+1) - (n-2*m)] {
                    self.u_max[k] += chain_length -2;
                    self.u_min[k] += chain_length;
                } else {
                    self.u_max[k] += chain_length;
                    self.u_min[k] += chain_length -2;
                }
            }
        }
    }

    fn set_bits(&mut self, left: bool, right: bool) {
        let mid = self.m;
        self.sequence.insert(mid, left);
        self.sequence.insert(mid + 1, right);
        self.m += 1;
    }

    fn update_fixed_energys(&mut self) {
        let m = self.m;
        let n = self.n;

        for k in 0..(n - m) {
            if m + k >= n - m {
                if self.sequence[m - 1] ^ self.sequence[3 * m + k - n] {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
            }

            if m >= k + 2 {
                if self.sequence[m - 1] ^ self.sequence[m - (k + 2)] {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
            }

            if n - m - k < m {
                if self.sequence[m] ^ self.sequence[n - m - (k + 1)] {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
            }

            if n - m + (k + 1) < n {
                if self.sequence[m] ^ self.sequence[m + (k + 1)] {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
            }
        }
    }

    fn update_u_min_max(&mut self) {

        let m = self.m;
        let n = self.n;

        self.u_max[n - m - 1] = 0;
        self.u_min[n - m - 1] = 0;

        for k in 0..(n - m - 1) {        //update both chains
            if m  <= k + 1 {                                                                              // m <= k check if free -> fixed
                let chain_length = ((n - (k+1) -m ) / (k+1)) + 1;

                if m-1 + chain_length*(k+1) == n-m {   //if double update
                    if chain_length == 1 {
                        self.u_max[k] -= 1;
                        self.u_min[k] -= 1;
                        continue;
                    } else if self.sequence[m-1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] { //to s_a != s_b
                        if chain_length % 2 == 0 {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2;
                            continue; 
                        } else {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 0;
                            continue;
                        }
                    } else if chain_length % 2 == 0 {  //to s_a = s_b even
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 0; 
                        continue;
                    } else {
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 2;
                        continue;
                    }
                }
            
                if m-1 + chain_length*(k+1) > n - m {   //from free to unfree
                    if chain_length == 1 {
                        self.u_max[k] -= 2;
                        self.u_min[k] -= 2;
                        continue;
                    } else if chain_length % 2 == 0 {
                        if self.sequence[m-1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] { //if goes to s_a != s_b
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 0;
                        }
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] { //if goes to s_a != s_b
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 0;
                        }
                        continue;


                    } else {
                        if self.sequence[m-1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] { //if goes to s_a != s_b
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 0; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 2;                  
                        }
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] { //if goes to s_a != s_b
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 0; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 2;                 
                        }
                        continue;
                    }
                }
            } else {                                                                                    //m > k Update chains.
                let chain_length = ( (n - 2*m)/(k+1) ) + 1;

                if m - 1 + chain_length*(k+1) == n-m {    //double update
                    if chain_length == 1 {
                        if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m) +(k+1)] { //length was 3
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3;
                            continue;
                        } else {
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1;
                            continue;
                        }
                    } else if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m) +(k+1)] { //from s_a != s_b
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] { //to s_a != s_b
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2;
                            continue;
                        } else { // to s_a = s_b
                            if chain_length % 2 == 0 {
                                self.u_max[k] -= 0;
                                self.u_min[k] -= 0;
                                continue;
                            } else {
                                self.u_max[k] -= 0;
                                self.u_min[k] -= 4;
                                continue;
                            }
                        }
                    } else {   // from s_a = s_b
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] { //to s_a != s_b
                            if chain_length % 2 == 0 {
                                self.u_max[k] -= 4;
                                self.u_min[k] -= 4;
                                continue;
                            } else {
                                self.u_max[k] -= 4;
                                self.u_min[k] -= 0;
                                continue;
                            }
                        } else { // to s_a = s_b
                            
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2;
                            continue;
                        }
                    }
                } else if chain_length == 1 {  //no double update
                    if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] { //if was s_a != s_b 
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 0;
                    } else {
                        self.u_max[k] -= 2;
                        self.u_min[k] -= 2;
                    }
                    if self.sequence[m + (k+1)] ^ self.sequence[n - m - chain_length*(k+1)] {  //if was s_a != s_b (was even length)
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 0;
                    } else {
                        self.u_max[k] -= 2;
                        self.u_min[k] -= 2;
                    }
                    continue;
                }


                if chain_length % 2 == 0 {   //even length

                    //first chain
                    if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {     //from s_a != s_b
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {         //to s_a != s_b
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3; 
                        } else {                     //to s_a = s_b
                            self.u_max[k] += 1;
                            self.u_min[k] -= 1; 
                        }
                    } else {                                                                           //from s_a = s_b
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {        //to s_a != s_b                                                //chain length even
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1; 
                        } else {                                                                       //to s_a = s_b
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1;
                        }
                    }

                    //second chain
                    if self.sequence[m + (k+1)] ^ self.sequence[n - m - chain_length*(k+1)] {     //from s_a != s_b
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {         //to s_a != s_b
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3; 
                        } else {                     //to s_a = s_b
                            self.u_max[k] += 1;
                            self.u_min[k] -= 1; 
                        }
                    } else {                                                                           //from s_a = s_b
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {        //to s_a != s_b                                        
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1; 
                        } else {                                                                       //to s_a = s_b
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1;
                        }
                    }
                    continue;

                } else {                                  //odd length

                    //first chain
                    if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {         //from s_a != s_b
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {  // to s_a != s_b                                                     //chain length even
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1; 
                        } else {                                                                       //to s_a = s_b
                            self.u_max[k] += 1;
                            self.u_min[k] -= 1;
                        }
                    } else {                                                                           //from s_a = s_b
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {     //to s_a != s_b
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1; 
                        } else {                                                                       //to s_a = s_b
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3;
                        }
                    }

                    //second chain
                    if self.sequence[m + (k+1)] ^ self.sequence[n - m - chain_length*(k+1)] {         //from s_a != s_b
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {  // to s_a != s_b                                         
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1; 
                        } else {                                                                       //to s_a = s_b
                            self.u_max[k] += 1;
                            self.u_min[k] -= 1;
                        }
                    } else {                                                                           //from s_a = s_b
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)]  {   //to s_a != s_b
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1; 
                        } else {                                                                       //to s_a = s_b
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3;
                        }
                    }
                    continue;
                }
            }  
        }
    }

    fn set_last_bit(&mut self, bit: bool) {
        let mid = self.sequence.len() / 2;
        self.sequence.insert(mid, bit);
        self.m += 1;
        let n = self.n;
        let m = self.m;

        for k in 0..(n - m) {
            if self.sequence[m - 1] ^ self.sequence[m - 2 - k] {
                self.fixed_energys[k] -= 1;
            } else {
                self.fixed_energys[k] += 1;
            }
        
            if self.sequence[m - 1] ^ self.sequence[m + k] {
                self.fixed_energys[k] -= 1;
            } else {
                self.fixed_energys[k] += 1;
            }
        }   
    }
}



fn branch(sequence: &mut BitSequence) -> BitSequence {
    
    let mut seq = sequence.clone();

    if !sequence.branched_mm {
        seq.set_bits(false, false);
        seq.branched_mm = false;
        seq.branched_mp = false;
        seq.branched_pm = false;
        seq.branched_pp = false;
        sequence.branched_mm = true;
        seq.update_fixed_energys();
        seq.update_u_min_max();

    } else if !sequence.branched_mp {
        seq.set_bits(false, true);
        seq.branched_mm = false;
        seq.branched_mp = false;
        seq.branched_pm = false;
        seq.branched_pp = false;
        sequence.branched_mp = true;
        seq.update_fixed_energys();
        seq.update_u_min_max();

    } else if !sequence.branched_pm {
        seq.set_bits(true, false);
        seq.branched_mm = false;
        seq.branched_mp = false;
        seq.branched_pm = false;
        seq.branched_pp = false;
        sequence.branched_pm = true;
        seq.update_fixed_energys();
        seq.update_u_min_max();

    } else {
        seq.set_bits(true, true);
        seq.branched_mm = false;
        seq.branched_mp = false;
        seq.branched_pm = false;
        seq.branched_pp = false;
        sequence.branched_pp = true;
        seq.update_fixed_energys();
        seq.update_u_min_max();
    }

    seq
}

fn set_max(individual: &mut BitSequence, k: usize) -> bool {
    let n = individual.n;
    let m = individual.m;
    let amount = if (k+1) <= n - 2*m { k+1 } else { n-2*m }; //the amount of chains

    for i in 0..amount {  //go through all chains. Determine which chains are fixed and then see,
                          //if the spins have been fixed differently before.
        let chain_length = ((n - 2*m + i)/(k+1) ) + 1;

       
        if individual.sequence[m-1-i] == individual.sequence[m-1-i + chain_length*(k+1) - (n-2*m)] {  
            if individual.sequence[m-1-i] {
                for j in 0..chain_length-1 {
                    if individual.free_spins_minus.contains(&(m-1-i + (j+1)*(k+1))) { 
                        return true;
                    } else {
                        individual.free_spins_plus.insert(m-1-i + (j+1)*(k+1));
                    }
                }
            } else {
                for j in 0..chain_length-1 {
                    if individual.free_spins_plus.contains(&(m-1-i + (j+1)*(k+1))) {
                        return true;
                    } else {
                        individual.free_spins_minus.insert(m-1-i + (j+1)*(k+1));
                    }
                }
            }
        }
    }
    false
}

fn set_min(individual: &mut BitSequence, k: usize) -> bool {
    let n = individual.n;
    let m = individual.m;
    let amount = if (k+1) <= n - 2*m { k+1 } else { n-2*m }; //the amount of chains

    for i in 0..amount {  //go through all chains. Determine which chains are fixed and then see,
                          //if the spins have been fixed differently before.
        let chain_length = ((n - 2*m + i)/(k+1) ) + 1;

        if (chain_length % 2 == 0 && individual.sequence[m-1-i] == individual.sequence[m-1-i + chain_length*(k+1) - (n-2*m)]) || 
        chain_length % 2 == 1 && individual.sequence[m-1-i] != individual.sequence[m-1-i + chain_length*(k+1) - (n-2*m)] {  
            if individual.sequence[m-1-i] {
                for j in 0..chain_length -1 {
                    if (j+1) % 2 == 0 {
                        if individual.free_spins_minus.contains(&(m-1-i + (j+1)*(k+1))) {
                            return true;
                        } else {
                            individual.free_spins_plus.insert(m-1-i + (j+1)*(k+1));
                        }
                    } else if individual.free_spins_plus.contains(&(m-1-i + (j+1)*(k+1))) {
                        return true;
                    } else {
                        individual.free_spins_minus.insert(m-1-i + (j+1)*(k+1));
                    }
                }
            } else {
                for j in 0..chain_length -1 {
                    if (j+1) % 2 == 0 {
                        if individual.free_spins_plus.contains(&(m-1-i + (j+1)*(k+1))) {
                            return true;
                        } else {
                            individual.free_spins_minus.insert(m-1-i + (j+1)*(k+1));
                        }
                    } else if individual.free_spins_minus.contains(&(m-1-i + (j+1)*(k+1))) {
                        return true;
                    } else {
                        individual.free_spins_plus.insert(m-1-i + (j+1)*(k+1));
                    }
                }
            }
        }
    }
    false
}

fn calculate_e_min(individual: &mut BitSequence, bound: isize, calls_set_min_max:&mut usize) -> isize {
    let m = individual.m;
    let mut result = vec![0; individual.n - 1];
    let mut bonus: isize = 0;
    let mut max = Vec::new();
    let mut min = Vec::new();
  
    for (k, item) in result.iter_mut().enumerate().take(individual.n - 1) {
        
        if individual.fixed_energys[k] <= -(individual.u_max[k] as isize) {
            *item = individual.fixed_energys[k] + individual.u_max[k] as isize;
            
            if m > (k+1) {
                max.push(k);
            }
        } else if individual.fixed_energys[k] >= individual.u_min[k] as isize {
            *item = individual.fixed_energys[k] - individual.u_min[k] as isize;
            
            if m > (k+1) {
                min.push(k);
            }
        } else {
            let g_k = if individual.m < (k+1) { 2 } else { 4 };
            *item = modular_min_abs(individual.fixed_energys[k] - individual.u_min[k] as isize, g_k as isize);
        }
    }
    let energy: isize = result.iter().map(|&x| x * x).sum();
    if energy <= bound && energy + 16 > bound {
        *calls_set_min_max += max.len() + min.len();
        for item in &max {
            bonus += if set_max(individual, *item) { 16 } else { 0 }; 
        } 
        for item in &min {
            bonus += if set_min(individual, *item) { 16 } else { 0 }; 
        } 
    }
    energy + bonus
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

    let two_m = 10;
    let mut roots = Vec::new();

    let time_roots = Instant::now();

    for i in 0..(1 << two_m) {
        let mut sequence = Vec::new();
        for j in 0..two_m {
            sequence.push((i & (1 << j)) != 0);
        }

        let mut individual = BitSequence::new(length);
        individual.n = length;
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
                individual.multi_update_fixed_energys();
                individual.multi_update_u_min_max();
                roots.push(individual);
            }
        } else {
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
            let mut ar = a.clone();
            reverse(&mut ar);
            let mut ac = a.clone();
            complement(&mut ac);

            let mut all_sequences = [
                individual.sequence.clone(),
                r.sequence.clone(),
                c.sequence.clone(),
                a.sequence.clone(),
                rc.sequence.clone(),
                ra.sequence.clone(),
                ar.sequence.clone(),
                ac.sequence.clone(),
            ];
        
            all_sequences.sort();
            let smallest_sequence = all_sequences.first().unwrap();

            if smallest_sequence == &individual.sequence {
                individual.initialize_fixed_energys();
                individual.multi_update_fixed_energys();
                individual.multi_update_u_min_max();
                roots.push(individual);
            }
        }
    }

    let duration_roots = time_roots.elapsed();
    println!("{} Roots created in {}", roots.len(), duration_roots.as_secs());

    let total_nodes = AtomicUsize::new(0);
    let total_calls_set_min_max = AtomicUsize::new(0);

    let time_total = Instant::now();

    roots.par_iter_mut().for_each(|root| {
        
        let mut individuals = vec![root.clone()];

        let mut nodes_visited = 0;

        let mut calls_set_min_max = 0;

        while let Some(mut individual) = individuals.pop() {
            nodes_visited += 1;

            let e_min = calculate_e_min(&mut individual, bound.load(Ordering::Relaxed), &mut calls_set_min_max);

            if e_min > bound.load(Ordering::Relaxed) {
                continue;
            } else if individual.m == individual.n / 2 {
                if individual.n % 2 != 0 {
                    let mut child_f = individual.clone();
                    child_f.set_last_bit(false);
                    let e_min = final_energy(&child_f);
                    if e_min <= bound.load(Ordering::Relaxed) {
                        let duration = time_total.elapsed().as_secs_f64();
                        println!("Sequence found in {} seconds {} nodes, with energy {}", duration, nodes_visited, e_min);
                        let _ = log_to_file(length, duration, nodes_visited, calls_set_min_max);
                        bound.store(e_min, Ordering::Relaxed);
                    }

                    let mut child_t = individual.clone();
                    child_t.set_last_bit(true);
                    let e_min = final_energy(&child_t);
                    if e_min <= bound.load(Ordering::Relaxed) {
                        let duration = time_total.elapsed().as_secs_f64();
                        println!("Sequence found in {} seconds {} nodes, with energy {}", duration, nodes_visited, e_min);
                        let _ = log_to_file(length, duration, nodes_visited, calls_set_min_max);
                        bound.store(e_min, Ordering::Relaxed);
                    }
                } else {
                    let e_min = calculate_e_min(&mut individual, bound.load(Ordering::Relaxed), &mut calls_set_min_max);
                    if e_min <= bound.load(Ordering::Relaxed) {
                        let duration = time_total.elapsed().as_secs_f64();
                        println!("Sequence found in {} seconds {} nodes, with energy {}", duration, nodes_visited, e_min);
                        let _ = log_to_file(length, duration, nodes_visited, calls_set_min_max);
                        bound.store(e_min, Ordering::Relaxed);
                    }
                }
            } else {
                let child = branch(&mut individual);
                individuals.push(child);
            }
            
            if !individual.branched_pp && individual.m < individual.n / 2 {
                if individuals.is_empty() {
                    individuals.push(individual);
                } else {
                    individuals.insert(individuals.len() - 1, individual);
                }
            }
        }
        total_nodes.fetch_add(nodes_visited, Ordering::Relaxed);
        total_calls_set_min_max.fetch_add(calls_set_min_max, Ordering::Relaxed);
    });
    let total_duration = time_total.elapsed().as_secs_f64();
    println!("Tree explored in {} and {} nodes", total_duration, total_nodes.load(Ordering::Relaxed));
    let _ = log_to_file(length, total_duration, total_nodes.load(Ordering::Relaxed), total_calls_set_min_max.load(Ordering::Relaxed));
}