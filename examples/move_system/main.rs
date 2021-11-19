use flecs_sys::*;
use std::ffi::{CStr, CString};
use std::mem::*;

use chumsky::{prelude::*, stream::Stream};
use ariadne::{Report, ReportKind, Label, Source, Color, Fmt};
use std::{collections::HashMap, env, fs, fmt};

use std::sync::{mpsc::channel, Arc};
use std::thread;
use std::time::Duration;
use std::str;

use anyhow::{anyhow, Result};
use atomic_refcell::AtomicRefCell;
use notify::{watcher, DebouncedEvent, RecursiveMode, Watcher};
use wasmtime::*;


pub type Span = std::ops::Range<usize>;

// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
// enum Token {
//     Null,
//     Bool(bool),
//     Num(String),
//     Str(String),
//     Op(String),
//     Ctrl(char),
//     Ident(String),
//     Fn,
//     Let,
//     Print,
//     If,
//     Else,
// }

// impl fmt::Display for Token {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         match self {
//             Token::Null => write!(f, "null"),
//             Token::Bool(x) => write!(f, "{}", x),
//             Token::Num(n) => write!(f, "{}", n),
//             Token::Str(s) => write!(f, "{}", s),
//             Token::Op(s) => write!(f, "{}", s),
//             Token::Ctrl(c) => write!(f, "{}", c),
//             Token::Ident(s) => write!(f, "{}", s),
//             Token::Fn => write!(f, "fn"),
//             Token::Let => write!(f, "let"),
//             Token::Print => write!(f, "print"),
//             Token::If => write!(f, "if"),
//             Token::Else => write!(f, "else"),
//         }
//     }
// }

// fn lexer() -> impl Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
//     // A parser for numbers
//     let num = text::int(10)
//         .chain::<char, _, _>(just('.').chain(text::digits(10)).or_not().flatten())
//         .collect::<String>()
//         .map(Token::Num);

//     // A parser for strings
//     let str_ = just('"')
//         .ignore_then(filter(|c| *c != '"').repeated())
//         .then_ignore(just('"'))
//         .collect::<String>()
//         .map(Token::Str);

//     // A parser for operators
//     let op = one_of("+-*/!=".chars())
//         .repeated().at_least(1)
//         .collect::<String>()
//         .map(Token::Op);

//     // A parser for control characters (delimiters, semicolons, etc.)
//     let ctrl = one_of("()[]{};,".chars()).map(|c| Token::Ctrl(c));

//     // A parser for identifiers and keywords
//     let ident = text::ident().map(|ident: String| match ident.as_str() {
//         "fn" => Token::Fn,
//         "let" => Token::Let,
//         "print" => Token::Print,
//         "if" => Token::If,
//         "else" => Token::Else,
//         "true" => Token::Bool(true),
//         "false" => Token::Bool(false),
//         "null" => Token::Null,
//         _ => Token::Ident(ident),
//     });

//     // A single token can be one of the above
//     let token = num
//         .or(str_)
//         .or(op)
//         .or(ctrl)
//         .or(ident)
//         .recover_with(skip_then_retry_until([]));

//     token
//         .map_with_span(|tok, span| (tok, span))
//         .padded()
//         .repeated()
// }

// #[derive(Clone, Debug, PartialEq)]
// enum Value {
//     Null,
//     Bool(bool),
//     Num(f64),
//     Str(String),
//     List(Vec<Value>),
//     Func(String),
// }

// impl Value {
//     fn num(self, span: Span) -> Result<f64, Error> {
//         if let Value::Num(x) = self {
//             Ok(x)
//         } else {
//             Err(Error {
//                 span,
//                 msg: format!("'{}' is not a number", self),
//             })
//         }
//     }
// }

// impl std::fmt::Display for Value {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         match self {
//             Self::Null => write!(f, "null"),
//             Self::Bool(x) => write!(f, "{}", x),
//             Self::Num(x) => write!(f, "{}", x),
//             Self::Str(x) => write!(f, "{}", x),
//             Self::List(xs) => write!(f, "[{}]", xs
//                 .iter()
//                 .map(|x| x.to_string())
//                 .collect::<Vec<_>>()
//                 .join(", ")),
//             Self::Func(name) => write!(f, "<function: {}>", name),
//         }
//     }
// }

// #[derive(Clone, Debug)]
// enum BinaryOp {
//     Add, Sub,
//     Mul, Div,
//     Eq, NotEq,
// }

// pub type Spanned<T> = (T, Span);

// // An expression node in the AST. Children are spanned so we can generate useful runtime errors.
// #[derive(Debug)]
// enum Expr {
//     Error,
//     Value(Value),
//     List(Vec<Spanned<Self>>),
//     Local(String),
//     Let(String, Box<Spanned<Self>>, Box<Spanned<Self>>),
//     Then(Box<Spanned<Self>>, Box<Spanned<Self>>),
//     Binary(Box<Spanned<Self>>, BinaryOp, Box<Spanned<Self>>),
//     Call(Box<Spanned<Self>>, Spanned<Vec<Spanned<Self>>>),
//     If(Box<Spanned<Self>>, Box<Spanned<Self>>, Box<Spanned<Self>>),
//     Print(Box<Spanned<Self>>),
// }

// // A function node in the AST.
// #[derive(Debug)]
// struct ScriptFunc {
//     args: Vec<String>,
//     body: Spanned<Expr>,
// }

// #[derive(Debug)]
// struct Struct {
//     fields: Vec<String>,
//     field_names: Vec<String>
// }

// fn expr_parser() -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone {
//     recursive(|expr| {
//         let raw_expr = recursive(|raw_expr| {
//             let val = filter_map(|span, tok| match tok {
//                 Token::Null => Ok(Expr::Value(Value::Null)),
//                 Token::Bool(x) => Ok(Expr::Value(Value::Bool(x))),
//                 Token::Num(n) => Ok(Expr::Value(Value::Num(n.parse().unwrap()))),
//                 Token::Str(s) => Ok(Expr::Value(Value::Str(s))),
//                 _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
//             })
//                 .labelled("value");

//             let ident = filter_map(|span, tok| match tok {
//                 Token::Ident(ident) => Ok(ident.clone()),
//                 _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
//             })
//                 .labelled("identifier");

//             // A list of expressions
//             let items = expr.clone()
//                 .chain(just(Token::Ctrl(',')).ignore_then(expr.clone()).repeated())
//                 .then_ignore(just(Token::Ctrl(',')).or_not())
//                 .or_not()
//                 .map(|item| item.unwrap_or_else(Vec::new));

//             // A let expression
//             let let_ = just(Token::Let)
//                 .ignore_then(ident)
//                 .then_ignore(just(Token::Op("=".to_string())))
//                 .then(raw_expr)
//                 .then_ignore(just(Token::Ctrl(';')))
//                 .then(expr.clone())
//                 .map(|((name, val), body)| Expr::Let(name, Box::new(val), Box::new(body)));

//             let list = items.clone()
//                 .delimited_by(Token::Ctrl('['), Token::Ctrl(']'))
//                 .map(Expr::List);

//             // 'Atoms' are expressions that contain no ambiguity
//             let atom = val
//                 .or(ident.map(Expr::Local))
//                 .or(let_)
//                 .or(list)
//                 // In Nano Rust, `print` is just a keyword, just like Python 2, for simplicity
//                 .or(just(Token::Print)
//                     .ignore_then(expr.clone().delimited_by(Token::Ctrl('('), Token::Ctrl(')')))
//                     .map(|expr| Expr::Print(Box::new(expr))))
//                 .map_with_span(|expr, span| (expr, span))
//                 // Atoms can also just be normal expressions, but surrounded with parentheses
//                 .or(expr.clone().delimited_by(Token::Ctrl('('), Token::Ctrl(')')))
//                 // Attempt to recover anything that looks like a parenthesised expression but contains errors
//                 .recover_with(nested_delimiters(
//                     Token::Ctrl('('), Token::Ctrl(')'),
//                     [
//                         (Token::Ctrl('['), Token::Ctrl(']')),
//                         (Token::Ctrl('{'), Token::Ctrl('}')),
//                     ],
//                     |span| (Expr::Error, span),
//                 ))
//                 // Attempt to recover anything that looks like a list but contains errors
//                 .recover_with(nested_delimiters(
//                     Token::Ctrl('['), Token::Ctrl(']'),
//                     [
//                         (Token::Ctrl('('), Token::Ctrl(')')),
//                         (Token::Ctrl('{'), Token::Ctrl('}')),
//                     ],
//                     |span| (Expr::Error, span),
//                 ));

//             // Function calls have very high precedence so we prioritise them
//             let call = atom.then(items
//                     .delimited_by(Token::Ctrl('('), Token::Ctrl(')'))
//                     .map_with_span(|args, span| (args, span))
//                     .repeated())
//                 .foldl(|f, args| {
//                     let span = f.1.start..args.1.end;
//                     (Expr::Call(Box::new(f), args), span)
//                 });

//             // Product ops (multiply and divide) have equal precedence
//             let op = just(Token::Op("*".to_string())).to(BinaryOp::Mul).or(just(Token::Op("/".to_string())).to(BinaryOp::Div));
//             let product = call.clone().then(op.then(call).repeated())
//                 .foldl(|a, (op, b)| {
//                     let span = a.1.start..b.1.end;
//                     (Expr::Binary(Box::new(a), op, Box::new(b)), span)
//                 });

//             // Sum ops (add and subtract) have equal precedence
//             let op = just(Token::Op("+".to_string())).to(BinaryOp::Add).or(just(Token::Op("-".to_string())).to(BinaryOp::Sub));
//             let sum = product.clone().then(op.then(product).repeated())
//                 .foldl(|a, (op, b)| {
//                     let span = a.1.start..b.1.end;
//                     (Expr::Binary(Box::new(a), op, Box::new(b)), span)
//                 });

//             // Comparison ops (equal, not-equal) have equal precedence
//             let op = just(Token::Op("==".to_string())).to(BinaryOp::Eq).or(just(Token::Op("!=".to_string())).to(BinaryOp::NotEq));
//             let compare = sum.clone().then(op.then(sum).repeated())
//                 .foldl(|a, (op, b)| {
//                     let span = a.1.start..b.1.end;
//                     (Expr::Binary(Box::new(a), op, Box::new(b)), span)
//                 });

//             compare
//         });

//         // Blocks are expressions but delimited with braces
//         let block = expr.clone()
//             .delimited_by(Token::Ctrl('{'), Token::Ctrl('}'))
//             // Attempt to recover anything that looks like a block but contains errors
//             .recover_with(nested_delimiters(
//                 Token::Ctrl('{'), Token::Ctrl('}'),
//                 [
//                     (Token::Ctrl('('), Token::Ctrl(')')),
//                     (Token::Ctrl('['), Token::Ctrl(']')),
//                 ],
//                 |span| (Expr::Error, span),
//             ));

//         let if_ = recursive(|if_| {
//             just(Token::If)
//                 .ignore_then(expr.clone())
//                 .then(block.clone())
//                 .then(just(Token::Else).ignore_then(block.clone().or(if_)).or_not())
//                 .map_with_span(|((cond, a), b), span| (
//                     Expr::If(Box::new(cond), Box::new(a), Box::new(match b {
//                         Some(b) => b,
//                         // If an `if` expression has no trailing `else` block, we magic up one that just produces null
//                         None => (Expr::Value(Value::Null), span.clone()),
//                     })),
//                     span,
//                 ))
//         });

//         // Both blocks and `if` are 'block expressions' and can appear in the place of statements
//         let block_expr = block
//             .or(if_)
//             .labelled("block");

//         let block_chain = block_expr.clone().then(block_expr.clone().repeated())
//             .foldl(|a, b| {
//                 let span = a.1.start..b.1.end;
//                 (Expr::Then(Box::new(a), Box::new(b)), span)
//             });

//         block_chain
//             // Expressions, chained by semicolons, are statements
//             .or(raw_expr.clone())
//             .then(just(Token::Ctrl(';')).ignore_then(expr.or_not()).repeated())
//             .foldl(|a, b| {
//                 let span = a.1.clone(); // TODO: Not correct
//                 (
//                     Expr::Then(Box::new(a), Box::new(match b {
//                         Some(b) => b,
//                         None => (Expr::Value(Value::Null), span.clone()),
//                     })),
//                     span,
//                 )
//             })
//     })
// }

// fn funcs_parser() -> impl Parser<Token, HashMap<String, ScriptFunc>, Error = Simple<Token>> + Clone {
//     let ident = filter_map(|span, tok| match tok {
//         Token::Ident(ident) => Ok(ident.clone()),
//         _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
//     });

//     // Argument lists are just identifiers separated by commas, surrounded by parentheses
//     let args = ident.clone()
//         .separated_by(just(Token::Ctrl(',')))
//         .allow_trailing()
//         .delimited_by(Token::Ctrl('('), Token::Ctrl(')'))
//         .labelled("function args");

//     let func = just(Token::Fn)
//         .ignore_then(ident
//             .map_with_span(|name, span| (name, span))
//             .labelled("function name"))
//         .then(args)
//         .then(expr_parser()
//             .delimited_by(Token::Ctrl('{'), Token::Ctrl('}'))
//             // Attempt to recover anything that looks like a function body but contains errors
//             .recover_with(nested_delimiters(
//                 Token::Ctrl('{'), Token::Ctrl('}'),
//                 [
//                     (Token::Ctrl('('), Token::Ctrl(')')),
//                     (Token::Ctrl('['), Token::Ctrl(']')),
//                 ],
//                 |span| (Expr::Error, span),
//             )))
//         .map(|((name, args), body)| (name, ScriptFunc { args, body }))
//         .labelled("function");

//     func
//         .repeated()
//         .try_map(|fs, _| {
//             let mut funcs = HashMap::new();
//             for ((name, name_span), f) in fs {
//                 if funcs.insert(name.clone(), f).is_some() {
//                     return Err(Simple::custom(name_span.clone(), format!("Function '{}' already exists", name)));
//                 }
//             }
//             Ok(funcs)
//         })
//         .then_ignore(end())
// }

// struct Error {
//     span: Span,
//     msg: String,
// }

// fn eval_expr(expr: &Spanned<Expr>, funcs: &HashMap<String, ScriptFunc>, stack: &mut Vec<(String, Value)>) -> Result<Value, Error> {
//     Ok(match &expr.0 {
//         Expr::Error => unreachable!(), // Error expressions only get created by parser errors, so cannot exist in a valid AST
//         Expr::Value(val) => val.clone(),
//         Expr::List(items) => Value::List(items
//             .iter()
//             .map(|item| eval_expr(item, funcs, stack))
//             .collect::<Result<_, _>>()?),
//         Expr::Local(name) => stack
//             .iter()
//             .rev()
//             .find(|(l, _)| l == name)
//             .map(|(_, v)| v.clone())
//             .or_else(|| Some(Value::Func(name.clone())).filter(|_| funcs.contains_key(name)))
//             .ok_or_else(|| Error {
//                 span: expr.1.clone(),
//                 msg: format!("No such variable '{}' in scope", name),
//             })?,
//         Expr::Let(local, val, body) => {
//             let val = eval_expr(val, funcs, stack)?;
//             stack.push((local.clone(), val));
//             let res = eval_expr(body, funcs, stack)?;
//             stack.pop();
//             res
//         },
//         Expr::Then(a, b) => {
//             eval_expr(a, funcs, stack)?;
//             eval_expr(b, funcs, stack)?
//         },
//         Expr::Binary(a, BinaryOp::Add, b) => Value::Num(eval_expr(a, funcs, stack)?.num(a.1.clone())? + eval_expr(b, funcs, stack)?.num(b.1.clone())?),
//         Expr::Binary(a, BinaryOp::Sub, b) => Value::Num(eval_expr(a, funcs, stack)?.num(a.1.clone())? - eval_expr(b, funcs, stack)?.num(b.1.clone())?),
//         Expr::Binary(a, BinaryOp::Mul, b) => Value::Num(eval_expr(a, funcs, stack)?.num(a.1.clone())? * eval_expr(b, funcs, stack)?.num(b.1.clone())?),
//         Expr::Binary(a, BinaryOp::Div, b) => Value::Num(eval_expr(a, funcs, stack)?.num(a.1.clone())? / eval_expr(b, funcs, stack)?.num(b.1.clone())?),
//         Expr::Binary(a, BinaryOp::Eq, b) => Value::Bool(eval_expr(a, funcs, stack)? == eval_expr(b, funcs, stack)?),
//         Expr::Binary(a, BinaryOp::NotEq, b) => Value::Bool(eval_expr(a, funcs, stack)? != eval_expr(b, funcs, stack)?),
//         Expr::Call(func, (args, args_span)) => {
//             let f = eval_expr(func, funcs, stack)?;
//             match f {
//                 Value::Func(name) => {
//                     let f = &funcs[&name];
//                     let mut stack = if f.args.len() != args.len() {
//                         return Err(Error {
//                             span: args_span.clone(),
//                             msg: format!("'{}' called with wrong number of arguments (expected {}, found {})", name, f.args.len(), args.len()),
//                         });
//                     } else {
//                         f.args
//                             .iter()
//                             .zip(args.iter())
//                             .map(|(name, arg)| Ok((name.clone(), eval_expr(arg, funcs, stack)?)))
//                             .collect::<Result<_, _>>()?
//                     };
//                     eval_expr(&f.body, funcs, &mut stack)?
//                 },
//                 f => return Err(Error {
//                     span: func.1.clone(),
//                     msg: format!("'{:?}' is not callable", f),
//                 }),
//             }
//         },
//         Expr::If(cond, a, b) => {
//             let c = eval_expr(cond, funcs, stack)?;
//             match c {
//                 Value::Bool(true) => eval_expr(a, funcs, stack)?,
//                 Value::Bool(false) => eval_expr(b, funcs, stack)?,
//                 c => return Err(Error {
//                     span: cond.1.clone(),
//                     msg: format!("Conditions must be booleans, found '{:?}'", c),
//                 }),
//             }
//         },
//         Expr::Print(a) => {
//             let val = eval_expr(a, funcs, stack)?;
//             println!("{}", val);
//             val
//         },
//     })
// }

// fn main() {
//     // let src = fs::read_to_string(env::args().nth(1).expect("Expected file argument")).expect("Failed to read file");
//     let src = r#"
// fn factorial(x) {
//     if x == 0 {
//         1
//     } else {
//         x * factorial(x - 1)
//     }
// }

// fn main() {
//     let three = 3;
//     let meaning_of_life = three * 14 + 1;

//     print("Hello, world!");
//     print("The meaning of life is...");

//     if meaning_of_life == 42 {
//         print(meaning_of_life);
//     } else {
//         print("...something we cannot know");

//         print("However, I can tell you that the factorial of 10 is...");
//         print(factorial(10, 11));
//     }
// }
//     "#;
//     let (tokens, mut errs) = lexer().parse_recovery(src);

//     let parse_errs = if let Some(tokens) = tokens {
//         // println!("Tokens = {:?}", tokens);
//         let len = src.chars().count();
//         let (ast, parse_errs) = funcs_parser().parse_recovery(Stream::from_iter(len..len + 1, tokens.into_iter()));

//         println!("{:#?}", ast);
//         if let Some(funcs) = ast.filter(|_| errs.len() + parse_errs.len() == 0) {
//             if let Some(main) = funcs.get("main") {
//                 assert_eq!(main.args.len(), 0);
//                 match eval_expr(&main.body, &funcs, &mut Vec::new()) {
//                     Ok(val) => println!("Return value: {}", val),
//                     Err(e) => errs.push(Simple::custom(e.span, e.msg)),
//                 }
//             } else {
//                 panic!("No main function!");
//             }
//         }

//         parse_errs
//     } else {
//         Vec::new()
//     };

//     errs
//         .into_iter()
//         .map(|e| e.map(|c| c.to_string()))
//         .chain(parse_errs
//             .into_iter()
//             .map(|e| e.map(|tok| tok.to_string())))
//         .for_each(|e| {
//             let report = Report::build(ReportKind::Error, (), e.span().start);

//             let report = match e.reason() {
//                 chumsky::error::SimpleReason::Unclosed { span, delimiter } => report
//                     .with_message(format!("Unclosed delimiter {}", delimiter.fg(Color::Yellow)))
//                     .with_label(Label::new(span.clone())
//                         .with_message(format!("Unclosed delimiter {}", delimiter.fg(Color::Yellow)))
//                     .with_color(Color::Yellow))
//                     .with_label(Label::new(e.span())
//                         .with_message(format!("Must be closed before this {}", e.found().unwrap_or(&"end of file".to_string()).fg(Color::Red)))
//                     .with_color(Color::Red)),
//                 chumsky::error::SimpleReason::Unexpected => report
//                     .with_message(format!("{}, expected {}", if e.found().is_some() {
//                         "Unexpected token in input"
//                     } else {
//                         "Unexpected end of input"
//                     }, if e.expected().len() == 0 {
//                         "end of input".to_string()
//                     } else {
//                         e.expected().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
//                     }))
//                     .with_label(Label::new(e.span())
//                         .with_message(format!("Unexpected token {}", e.found().unwrap_or(&"end of file".to_string()).fg(Color::Red)))
//                     .with_color(Color::Red)),
//                 chumsky::error::SimpleReason::Custom(msg) => report
//                     .with_message(msg)
//                     .with_label(Label::new(e.span())
//                         .with_message(format!("{}", msg.fg(Color::Red)))
//                     .with_color(Color::Red)),
//             };

//             report
//                 .finish()
//                 .print(Source::from(&src))
//                 .unwrap();
//         });
// }



#[repr(C)]
struct Position {
    x: f32,
    y: f32,
}

#[repr(C)]
struct Velocity {
    x: f32,
    y: f32,
}

unsafe extern "C" fn move_sys(it: *mut ecs_iter_t) {
    let count = (*it).count as usize;

    let posc = ecs_column_w_size(it, size_of::<Position>(), 1);
    let velc = ecs_column_w_size(it, size_of::<Velocity>(), 2);

    let p = std::ptr::slice_from_raw_parts_mut(posc as *mut Position, count)
        .as_mut()
        .unwrap();

    let v = std::ptr::slice_from_raw_parts_mut(velc as *mut Velocity, count)
        .as_mut()
        .unwrap();

    for i in 0..count {
        let position = p.get_mut(i).unwrap();
        let velocity = v.get_mut(i).unwrap();

        position.x += velocity.x * (*it).delta_time;
        position.y += velocity.y * (*it).delta_time;

        let ents = std::ptr::slice_from_raw_parts((*it).entities, count);
        let e_name = CStr::from_ptr(ecs_get_name((*it).world, (&*ents)[i]));

        println!(
            "{} moved to {{.x = {}, .y = {}}}",
            e_name.to_string_lossy().into_owned(),
            position.x,
            position.y
        );
    }
}

use std::mem;

struct FlecsSystem<T> {
    ptr: *mut ecs_iter_t,
    wasm_fn: Option<WASMSystemFn>,
    store: Option<Store<T>>,
    name: String,
    query: String,
    id: u64,
}

type FlecsSystemFn = unsafe extern fn(*mut ecs_iter_t);

type WASMSystemFn = TypedFunc<(i64), ()>;


impl<T> FlecsSystem<T> {
	fn new() -> FlecsSystem<T> {
		// initialize ffi_Object here
		FlecsSystem { ptr: std::ptr::null_mut(), wasm_fn: None, store: None, 
            name: "uninitialized".to_string(), query: "uninitialized".to_string(), id: 0}
	}
	fn from_ptr(ptr: *mut ecs_iter_t, ) -> FlecsSystem<T> {
		FlecsSystem { ptr: ptr, wasm_fn: None, store: None,
            name: "uninitialized".to_string(), query: "uninitialized".to_string(), id: 0}
	}

    fn set_wasm_fn(&mut self, wasm_fn: WASMSystemFn) {
		self.wasm_fn = Some(wasm_fn);
	}

    fn set_store(&mut self, store: Store<T>) {
        self.store = Some(store);
    }
	fn push_callback(&mut self, cb: FlecsSystemFn, world: *mut ecs_world_t) {
        unsafe {
            self.id = ecs_new_system(
                world,
                0,
                CString::new(self.name.clone()).unwrap().as_ptr(),
                EcsOnUpdate as u64,
                CString::new(self.query.clone()).unwrap().as_ptr(),
                Some(cb),
            );
        }

		// ffi_push_callback(self.ptr, cb);
        // ecs new system
	}

    fn call_wasm_fn(&mut self) {
        self.wasm_fn.unwrap().call(self.store.as_mut().unwrap(), self.ptr as i64);
    }
}


// impl Drop for FlecsSystem {
// 	fn drop(&mut self) {
// 		// destroy ffi_Object here
// 	}
// }

// The Solution
fn wrap_callback<T, F: Fn(&mut FlecsSystem<T>)>(_: F) -> FlecsSystemFn {
	assert!(mem::size_of::<F>() == 0);

	unsafe extern fn wrapped<T, F: Fn(&mut FlecsSystem<T>)>(ptr: *mut ecs_iter_t) {
		let mut object = FlecsSystem::from_ptr(ptr);
		let result = mem::transmute::<_, &F>(&())(&mut object);
		mem::forget(object);
		result
	}

	wrapped::<T, F>
}

// Usage
fn wasm_system_callback(system: &mut FlecsSystem<World>) {
	system.call_wasm_fn();
}


/// Path to the Compiled WASM File from assembly script.
const WASM_FILE_PATH: &str = "./build/optimized.wasm";

// https://stackoverflow.com/questions/49604196/arbitrary-bytes-in-webassembly-data-section


/// WASM Module Container is a simple wrapper around the [`Module`] from `wasmtime` crate so it could be shared between threads easily
/// it also contains the Engine that would be used to compile the module.
///
/// It is very cheap to clone the container and be shared between the threads.
#[derive(Clone)]
struct WasmModuleContainer {
    module: Arc<AtomicRefCell<Module>>,
    engine: Engine,
}

// impl WasmModuleContainer {
//     /// Calling Init will create a new WASM [`Engine`] and using that [`Engine`] to compile the Wasm module
//     /// This should only called once at the starting of the program.
//     pub fn init() -> Result<Self> {
//         let engine = Engine::default();
//         // let module = Module::from_file(&engine, WASM_FILE_PATH)?;
//         let module = Module::new(&engine, WASM_WAT)?;
//         Ok(Self {
//             engine,
//             module: Arc::new(AtomicRefCell::new(module)),
//         })
//     }

//     /// Create an [`Instance`] from the already compiled WASM [`Module`].
//     pub fn instance(&self) -> Result<Instance> {
//         let module = self.module.borrow();
//         let store = Store::new(&self.engine);
//         Instance::new(&store, &module, &[])
//     }

//     /// Reload and Recompile the WASM [`Module`].
//     /// Next Calls to [`WasmModuleContainer::instance`] will get a new [`Instance`] with the new compiled code.
//     pub fn reload(&self) -> Result<()> {
//         println!("Code Changed, Recompile..");
//         let module = Module::from_file(&self.engine, WASM_FILE_PATH)?;
//         *self.module.borrow_mut() = module;
//         println!("Hot Reloaded");
//         Ok(())
//     }
// }

fn host_print_i64(mut caller: Caller<'_, &World>, i: i64) {
    println!("host_print_i64 {}", i);
}

fn host_new_entity(mut caller: Caller<'_, &World>) -> i64 {
    let world = caller.data_mut();
    let id = world.new_id();
    println!("New id {} from WebAssembly", id);
    return id as i64;
}


fn host_new_system(
    mut caller: Caller<'_, &World>,
    sys_name_ptr: i32, 
    sys_name_len: i32,
    query_ptr: i32,
    query_len: i32) 
    -> i64 {
    let ecs = caller.data_mut().ecs.clone();

    let mem = match caller.get_export("memory") {
        Some(Extern::Memory(mem)) => mem,
        _ => {
            // Err(Trap::new("failed to find host memory"));
            return 0;
        },
    };

    // system name string
    let data = mem.data(&caller)
        .get(sys_name_ptr as u32 as usize..)
        .and_then(|arr| arr.get(..sys_name_len as u32 as usize));
    let sys_name_string = match data {
        Some(data) => match str::from_utf8(data) {
            Ok(s) => s,
            Err(_) => {
                // Err(Trap::new("invalid utf-8"));
                return 0;
            },
        },
        None => {
            // Err(Trap::new("pointer/length out of bounds"));
            return 0;
        },
    };
    
    let sys_name_string = sys_name_string.to_string();
    
    let data = mem.data(&caller)
        .get(sys_name_ptr as u32 as usize..)
        .and_then(|arr| arr.get(..sys_name_len as u32 as usize));
    let query_string = match data {
        Some(data) => match str::from_utf8(data) {
            Ok(s) => s,
            Err(_) => {
                // Err(Trap::new("invalid utf-8"));
                return 0;
            },
        },
        None => {
            // Err(Trap::new("pointer/length out of bounds"));
            return 0;
        },
    };
    let query_string = query_string.to_string();

    // let system_fn = wasm_system_func::<&World>("system_test".to_string(), 
    // &caller.data().instance.unwrap(), caller.as_context_mut().into());

    let system_fn = caller.get_export(&sys_name_string).unwrap();
    let system_fn = system_fn.into_func().unwrap()
                                                       .typed::<(i64), (), _>(&caller).unwrap();
    let mut flecs_system: FlecsSystem<World> = FlecsSystem::new();
    flecs_system.name = sys_name_string.to_string();
    flecs_system.query = query_string.to_string();
    flecs_system.wasm_fn = Some(system_fn);
    let wrapped = wrap_callback(wasm_system_callback);
    flecs_system.push_callback(wrapped, ecs);

    println!("host_new_system sys_name_ptr {} {} query_ptr {} {} id {}\n {} {:?}", 
    sys_name_ptr, sys_name_len, query_ptr,  query_len, flecs_system.id, sys_name_string, query_string);

    return flecs_system.id as i64;
}

fn host_new_component(
    mut caller: Caller<'_, &World>,
    comp_ptr: i32, 
    comp_len: i32,
    str_ptr: i32,
    str_len: i32) 
    -> i64 {
    let ecs = caller.data_mut().ecs.clone();

    let mem = match caller.get_export("memory") {
        Some(Extern::Memory(mem)) => mem,
        _ => {
            // Err(Trap::new("failed to find host memory"));
            return 0;
        },
    };

    // component string
    let data = mem.data(&caller)
        .get(str_ptr as u32 as usize..)
        .and_then(|arr| arr.get(..str_len as u32 as usize));
    let comp_string = match data {
        Some(data) => match str::from_utf8(data) {
            Ok(s) => s,
            Err(_) => {
                // Err(Trap::new("invalid utf-8"));
                return 0;
            },
        },
        None => {
            // Err(Trap::new("pointer/length out of bounds"));
            return 0;
        },
    };
    
    let comp_string = comp_string.clone();
    // component data
    let comp_data = mem.data(&caller)
        .get(comp_ptr as u32 as usize..)
        .and_then(|arr| arr.get(..comp_len as u32 as usize));
    let name = CString::new(comp_string).unwrap().as_ptr();
    unsafe {
        let mut id = 0;
        {
            id = ecs_new_component(
                ecs,
                0,
                name,
                unsafe {std::mem::size_of_val(&comp_data) },
                unsafe {std::mem::align_of_val(&comp_data)},
            );
        }

        println!("host_new_component comp_ptr {} {} str_ptr {} {} id {}\n {} {:?}", 
        comp_ptr, comp_len, str_ptr, str_len, id, comp_string, comp_data);

        return id as i64;
    }; 
}

fn indirect_func<T>(name: String, instance: &Instance, mut store: &mut Store<T>) -> TypedFunc<(), (i64)> {
    instance.get_typed_func::<(), (i64), _>(&mut store, &name).unwrap()
}

fn wasm_system_func<T>(name: String, instance: &Instance, mut store: &mut Store<T>) -> TypedFunc<(i64), ()> {
    instance.get_typed_func::<(i64), (), _>(&mut store, &name).unwrap()
}

const WASM_WAT: &str = r#"
(module
    (import "host" "hello" (func $host_new_id_fn (result i64)))
    (import "host" "hello" (func $host_new_component_fn (param i32 i32 i32 i32) (result i64)))
    (import "host" "hello" (func $host_print_i64_fn (param i64)))
    (import "host" "hello" (func $host_new_system_fn (param i32 i32 i32 i32) (result i64)))

    (func (export "hello") (result i64)
        call $host_new_id_fn
        )
    (func (export "component_test") (result i64)
        i32.const 0 ;; comp_ptr
        i32.const 8 ;; comp_len
        i32.const 8 ;; str_ptr
        i32.const 4 ;; str_len
        call $host_new_component_fn
        )

    (func (export "fred") (param i64)
        local.get 0
        call $host_print_i64_fn ;; this should print the value of the pointer to ecs_iter_t
    )

    (func $new_system_test (result i64)
        i32.const 12 ;; sys_name_ptr
        i32.const 4 ;; sys_name_len
        i32.const 8 ;; query_ptr 
        i32.const 4 ;; query_len
        call $host_new_system_fn
        )

    (func (export "main") 
        call $new_system_test
        call $host_print_i64_fn 
    )

    (memory (export "memory") 1)
    (data (i32.const 0) "\04\02")
    (data (i32.const 4) "\06\09")
    (data (i32.const 8) "Test")
    (data (i32.const 12) "fred")

)
"#;

fn main() -> Result<()> {
    // let container = WasmModuleContainer::init()?;
    // let container2 = container.clone();
    // // Start the watching for changes thread.
    // thread::spawn(move || watch_for_changes(container2));


    let mut world = World::new();

    let engine = Engine::default();
    let module = Module::new(&engine, WASM_WAT)?;

    let mut store = Store::new(&engine, &world);
    // let host_hello = Func::wrap(&mut store, |caller: Caller<'_, u32>, param: i32| {
    //     println!("Got {} from WebAssembly", param);
    //     println!("my host state is: {}", caller.data());
    // });

    let host_new_id_fn = Func::wrap(&mut store, host_new_entity);
    let host_new_component_fn = Func::wrap(&mut store, host_new_component);
    let host_print_i64_fn = Func::wrap(&mut store, host_print_i64);
    let host_new_system_fn = Func::wrap(&mut store, host_new_system);

    // Instantiation of a module requires specifying its imports and then
    // afterwards we can fetch exports by name, as well as asserting the
    // type signature of the function with `get_typed_func`.
    let instance = Instance::new(&mut store, &module, &[
            host_new_id_fn.into(), 
            host_new_component_fn.into(),
            host_print_i64_fn.into(),
            host_new_system_fn.into(),
        ])?;

    // world.instance = Some(&instance);

    let hello = instance.get_typed_func::<(), (i64), _>(&mut store, "hello")?;
    let component_test = instance.get_typed_func::<(), (i64), _>(&mut store, "component_test")?;

    let component_test = indirect_func::<&World>("component_test".to_string(), &instance, &mut store);
    //let system_test = wasm_system_func::<&World>("system_test".to_string(), &instance, &mut store);

    let main_fn = instance.get_typed_func::<(), (), _>(&mut store, "main")?;

    // And finally we can call the wasm!
    let x = hello.call(&mut store, ())?;
    println!("back in rust ecs id {}", x);
    let x = component_test.call(&mut store, ())?;
    println!("rust component id {}", x);

    main_fn.call(&mut store, ())?;

    unsafe {
        ecs_progress(world.ecs, 0.0);
        ecs_progress(world.ecs, 0.0);
        ecs_progress(world.ecs, 0.0);
    }

    // let mut flecs_system: FlecsSystem<World> = FlecsSystem::new();
    // let wrapped = wrap_callback(wasm_system_callback);
    // flecs_system.push_callback(wrapped, world.ecs.clone());


    Ok(())

    // Main Application Loop.
    // loop {
    //     let instance = container.instance()?;
    //     let add = instance
    //         .get_func("add")
    //         .ok_or_else(|| anyhow!("No function named `add` found in wasm module"))?
    //         .get2::<i32, i32, i32>()?;

    //     println!("40 + 2 = {}", add(40, 2)?);

    //     thread::sleep(Duration::from_millis(200));
    // }
}

// fn watch_for_changes(container: WasmModuleContainer) -> Result<()> {
//     let (tx, rx) = channel();
//     let mut watcher = watcher(tx, Duration::from_millis(10))?;
//     watcher.watch(WASM_FILE_PATH, RecursiveMode::NonRecursive)?;
//     while let Ok(event) = rx.recv() {
//         match event {
//             DebouncedEvent::Create(_) | DebouncedEvent::Write(_) => {
//                 container.reload()?;
//             }
//             _ => {}
//         }
//     }
//     Ok(())
// }

struct World {
    ecs: *mut ecs_world_t,
    // instance: Option<&'a Instance>,

}

impl World {
    fn new() -> World {
		// initialize ffi_Object here
		unsafe { World { ecs: ecs_init()  } }
	}

    fn new_id(&self) -> ecs_entity_t {
        unsafe { ecs_new_id(self.ecs) }
    }

    // fn new_component(&self) -> ecs_entity_t {
    //     unsafe {

    //     }
    // }
}

fn test_ecs() {
    unsafe {
        let world = ecs_init();

        let pos_name = CString::new("Position").unwrap();
        let pos_id = ecs_new_component(
            world,
            0,
            pos_name.as_ptr(),
            size_of::<Position>(),
            align_of::<Position>(),
        );

        let vel_name = CString::new("Velocity").unwrap();
        let vel_id = ecs_new_component(
            world,
            0,
            vel_name.as_ptr(),
            size_of::<Velocity>(),
            align_of::<Velocity>(),
        );

        let comps = CString::new("Position, Velocity").unwrap();
        let sys_name = CString::new("Move").unwrap();

        ecs_new_system(
            world,
            0,
            sys_name.as_ptr(),
            EcsOnUpdate as u64,
            comps.as_ptr(),
            Some(move_sys),
        );

        let ename = CString::new("MyEntity").unwrap();
        let entity = ecs_new_entity(world, 0, ename.as_ptr(), comps.as_ptr());

        let mut p = Position { x: 0.0, y: 0.0 };
        let p_ptr: *mut std::ffi::c_void = &mut p as *mut _ as *mut std::ffi::c_void;
        ecs_set_ptr_w_id(world, entity, pos_id, size_of::<Position>(), p_ptr);

        let mut v = Velocity { x: 1.0, y: 1.0 };
        let v_ptr: *mut std::ffi::c_void = &mut v as *mut _ as *mut std::ffi::c_void;
        ecs_set_ptr_w_id(world, entity, vel_id, size_of::<Velocity>(), v_ptr);

        println!("Application move_system is running, press CTRL+C to exit...");

        while ecs_progress(world, 0.0) {}

        ecs_fini(world);
    }
}
