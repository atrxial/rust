fn fibonacci(n: u32) -> u32 {
    let mut arr = vec![0; (n + 1) as usize];
    arr[1] = 1;

    for i in 2..=n {
        arr[i as usize] = arr[(i - 1) as usize] + arr[(i - 2) as usize];
    }

    arr[n as usize]
}

fn main() {
    let n = 6;
    let result = fibonacci(n);
    println!("Fibonacci number at position {} is {}", n, result);
}

