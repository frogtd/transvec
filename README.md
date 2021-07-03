# transvec

transvec is a crate for transmuting Vecs without copying.

## Installation

Add this to your Cargo.toml:

```toml
[dependencies]
transvec = "0.1.0"
```

## Usage

```rust
use transvec::transmute_vec;
let input: Vec<u8> = vec![1, 2, 3, 4];
let output: Vec<u16, _> = transmute_vec(input).unwrap();
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licenses
- [MIT](https://choosealicense.com/licenses/mit/)
- [Unlicense](https://choosealicense.com/licenses/unlicense/)