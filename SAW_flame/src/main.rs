use std::time::{Instant, Duration};
use std::collections::HashSet;
use rayon::prelude::*;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use rand::Rng;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::hash::RandomState;


fn log_to_file(length: usize, time_taken: f64, nodes_visited: usize) -> std::io::Result<()> {
    // Open the file in append mode (or create it if it doesn't exist)
    let mut file = OpenOptions::new()
        .create(true)   // Create the file if it doesn't exist
        .append(true)   // Append to the file instead of overwriting
        .open("saw_symmetry.txt")?;

    // Write the data to the file
    writeln!(
        file,
        "N: {}, Time: {:.2} seconds, Nodes: {}",
        length, time_taken, nodes_visited
    )?;

    Ok(())
}

fn log_to_file_best_found(length: usize, time_taken: f64, best_energy: isize) -> std::io::Result<()> {
    // Open the file in append mode (or create it if it doesn't exist)
    let mut file = OpenOptions::new()
        .create(true)   // Create the file if it doesn't exist
        .append(true)   // Append to the file instead of overwriting
        .open("saw_parallel.txt")?;

    // Write the data to the file
    writeln!(
        file,
        "N: {}, Time: {:.2} seconds, Energy: {}",
        length, time_taken, best_energy
    )?;

    Ok(())
}

#[derive(Clone, Debug)]
struct BitSequence {
    sequence: u128,
    fixed_energys: Vec<isize>,
    energy: isize,
    visited: HashSet<u128 ,RandomState>,
    n: usize,
}
impl BitSequence {
    fn new(length: usize) -> Self {
        BitSequence {
            sequence: 0u128,
            fixed_energys: Vec::new(),
            energy: 0,
            visited: HashSet::with_hasher(RandomState::new()),
            n: length,
        }
    }

    fn initialize_random_sequence(&mut self, two_m: usize) {
        let mut rng = rand::rng();

        let middle_start = two_m / 2;
        let middle_end = self.n - (two_m / 2);

        // Flip random bits in the middle
        for bit in middle_start..middle_end {
            let random_bit = rng.random_bool(0.5);
            if random_bit {
                self.sequence |= 1 << bit; // set bit to 1
            } else {
                self.sequence &= !(1 << bit); // set bit to 0
            }
        }
    }

    fn initialize_fixed_energys(&mut self) {

        self.fixed_energys = Vec::with_capacity(self.n - 1);
        for _ in 0..self.n - 1 {
            self.fixed_energys.push(0);
        }

        for k in 0..(self.n - 1) {
            let shift = k + 1;
            for i in 0..(self.n - shift) {
                let a = (self.sequence >> i) & 1;
                let b = (self.sequence >> (i + shift)) & 1;
        
                if a ^ b == 1 {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
            }
        }
    }

    fn initialize_energy(&mut self) {
        for k in 0..self.n-1 {
            self.energy += self.fixed_energys[k]*self.fixed_energys[k];
        }
    }
}

fn neighbour_energy(sequence: u128, fixed_energys: Vec<isize>, n: usize, spin: usize) -> isize {
    let mut energy: isize = 0;

    for k in(0..n-1) {
        let mut ck = fixed_energys[k];
        let shift = k + 1;

        let spin_bit = (sequence >> spin) & 1;

        // Check left neighbor (spin - shift)
        if spin >= shift {
            let left_bit = (sequence >> (spin - shift)) & 1;
            if (spin_bit ^ left_bit) == 1 {
                ck += 2;
            } else {
                ck -= 2;
            }
        }

        // Check right neighbor (spin + shift)
        if spin + shift <= n - 1 {
            let right_bit = (sequence >> (spin + shift)) & 1;
            if (spin_bit ^ right_bit) == 1 {
                ck += 2;
            } else {
                ck -= 2;
            }
        }

        energy += ck * ck
    }
    energy
}

fn best_unvisited(individual: &BitSequence, two_m: &usize) -> (usize, isize) {
    let center_spins: Vec<usize> = (*two_m / 2..individual.n - *two_m / 2)
        .filter(|&spin| {
            let flipped = individual.sequence ^ (1 << spin);
            !individual.visited.contains(&flipped)
        })
        .collect();

    center_spins
        .into_par_iter()
        .map(|spin| {

            // Clone with flipped bit for energy calc
            let tmp_sequence = individual.sequence;
            let tmp_fixed_energys = individual.fixed_energys.clone();
            let tmp_n = individual.n;

            let energy = neighbour_energy(tmp_sequence, tmp_fixed_energys, tmp_n, spin);
            (spin, energy)
        })
        .reduce_with(|a, b| if a.1 < b.1 { a } else { b })
        .unwrap_or((individual.n + 1, isize::MAX))
}


fn update(individual: &mut BitSequence, best_energy: &mut isize, best_sequence: &mut u128, two_m: &usize) {

    let (spin, energy) = best_unvisited(individual, two_m);

    individual.sequence ^= 1 << spin;

    // Insert updated sequence into visited
    individual.visited.insert(individual.sequence);

    // Update energy
    individual.energy = energy;

    // Update fixed_energys based on flipped bit at 'spin'
    for k in 0..(individual.n - 1) {
        let shift = k + 1;

        if spin >= shift {
            let left_bit = (individual.sequence >> (spin - shift)) & 1;
            let spin_bit = (individual.sequence >> spin) & 1;
            if (spin_bit ^ left_bit) == 1 {
                individual.fixed_energys[k] -= 2;
            } else {
                individual.fixed_energys[k] += 2;
            }
        }

        if spin + shift <= individual.n - 1 {
            let right_bit = (individual.sequence >> (spin + shift)) & 1;
            let spin_bit = (individual.sequence >> spin) & 1;
            if (spin_bit ^ right_bit) == 1 {
                individual.fixed_energys[k] -= 2;
            } else {
                individual.fixed_energys[k] += 2;
            }
        }
    }

    if energy < *best_energy {
        *best_energy = energy;
        *best_sequence = individual.sequence;
    }
}

fn reverse(individual: &mut BitSequence) {
    let mut reversed = 0u128;
    for i in 0..individual.n {
        // Extract bit i
        let bit = (individual.sequence >> i) & 1;
        // Set bit at reversed position
        reversed |= bit << (individual.n - 1 - i);
    }
    individual.sequence = reversed;
}


fn complement(individual: &mut BitSequence) {
    // Flip all bits in the sequence within its length `n`
    // Create a mask with `n` 1-bits, so we only flip bits within valid length
    let mask = if individual.n == 128 {
        u128::MAX
    } else {
        (1u128 << individual.n) - 1
    };

    individual.sequence = (!individual.sequence) & mask;
}


fn alternate_complement(individual: &mut BitSequence) {
    let mid = individual.n / 2;
    let parity = individual.n % 2;

    // Flip bits at even indices in [0..mid)
    for i in (0..mid).step_by(2) {
        individual.sequence ^= 1 << i;
    }

    // Flip bits at indices in [mid..2*mid), 
    // but counted backward as (2*mid - 1 - i), where i % 2 != parity
    for i in 0..mid {
        if i % 2 != parity {
            let pos = 2 * mid - 1 - i;
            individual.sequence ^= 1 << pos;
        }
    }
}



fn main() {

    let args: Vec<String> = env::args().collect();

    let length: usize = args.get(1)
        .expect("Provide N")
        .parse()
        .expect("N as usize");
    let known_best: isize = args.get(2)
        .expect("Provide bound")
        .parse()
        .expect("bound as isize");

    let stop_flag = Arc::new(AtomicBool::new(false));
    let time_total = Instant::now();
    let best_found = Arc::new(AtomicIsize::new(isize::MAX));
    let max_duration = Duration::from_secs(10);

    let two_m = 6;
    let mut roots = Vec::new();

    let time_roots = Instant::now();

    for i in 0..(1u128 << two_m) {
        let mut individual = BitSequence::new(length);
        individual.sequence = i;

        for j in (0..two_m / 2) {
            let a = (individual.sequence >> j + two_m /2) & 1;
            individual.sequence &= !(1 << j + two_m /2);
            if a == 1 {
                individual.sequence |= 1 << (individual.n - two_m /2 + j);
            }
        }

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
                individual.sequence,
                r.sequence,
                c.sequence,
                a.sequence,
                rc.sequence,
                ra.sequence,
                ca.sequence,
                rca.sequence,
            ];

            all_sequences.sort();
            let smallest_sequence = all_sequences.first().unwrap();

            if smallest_sequence == &individual.sequence {
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
                individual.sequence,
                r.sequence,
                c.sequence,
                a.sequence,
                rc.sequence,
                ra.sequence,
                ar.sequence,
                ac.sequence,
            ];

            all_sequences.sort();
            let smallest_sequence = all_sequences.first().unwrap();

            if smallest_sequence == &individual.sequence {
                roots.push(individual);
            }
        }
    }
    let duration_roots = time_roots.elapsed();
    println!("{} Roots created in {} seconds", roots.len(), duration_roots.as_secs());
    for root in &roots {
        for i in (0..root.n).rev() {
            let bit = (root.sequence >> i) & 1;
            print!("{}", bit);
        }
        println!();
    }
    

    roots.par_iter_mut().for_each(|root| {

        let individual = root;
        individual.initialize_random_sequence(two_m);
        individual.initialize_fixed_energys();
        individual.initialize_energy();

        let mut best_energy = individual.energy;
        let mut best_sequence = individual.sequence;

        let mut saw_steps = 0;
        let time_single_saw = Instant::now();

        while !stop_flag.load(Ordering::Relaxed) && time_single_saw.elapsed() < max_duration {

            saw_steps += 1;
            update(individual, &mut best_energy, &mut best_sequence, &two_m);

            if saw_steps % 10000000 == 0 {
                individual.visited.clear();
            }

            if best_energy <= known_best {
                let duration = time_single_saw.elapsed().as_secs_f64();
                println!("Best Energy {} found in {} steps and {} seconds", best_energy, saw_steps, duration);
                let _ = log_to_file(length, duration, saw_steps);
                stop_flag.store(true, Ordering::Relaxed);
                return;
            }
        }
        if best_energy < best_found.load(Ordering::Relaxed) {
            best_found.fetch_min(best_energy, Ordering::Relaxed);
        }
    });
    if !stop_flag.load(Ordering::Relaxed) {
        let duration = time_total.elapsed().as_secs_f64();
        println!("Could not find best energy {}, found {}", known_best, best_found.load(Ordering::Relaxed));
        let _ = log_to_file_best_found(length, duration, best_found.load(Ordering::Relaxed));
    }
    let duration = time_total.elapsed().as_secs_f64();
    println!("All processes finished in {}", duration);

}

