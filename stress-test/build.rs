fn main() {
    println!("cargo:rustc-link-search=native=/usr/local/libtorch/lib");
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=torch_cuda");
    println!("cargo:rustc-link-lib=dylib=c10");
    //println!("cargo:rustc-link-lib=dylib=stdc++");
}
