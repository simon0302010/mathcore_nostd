#![no_std]

//! MathCore - symbolic math library for Rust
//!
//! basically a CAS (computer algebra system) that can do symbolic math,
//! solve equations, differentiate, integrate, etc.
//!
//! ```rust
//! use mathcore_nostd::MathCore;
//!
//! let math = MathCore::new();
//!
//! // basic stuff
//! let result = math.calculate("2 + 3 * 4").unwrap();
//! assert_eq!(result, 14.0);
//!
//! // calculus
//! let derivative = MathCore::differentiate("x^2", "x").unwrap();
//!
//! // solve equations
//! let roots = MathCore::solve("x^2 - 4", "x").unwrap();
//! ```

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod calculus;
pub mod differential;
pub mod engine;
pub mod matrix;
pub mod ml;
pub mod parser;
pub mod precision;
pub mod solver;
pub mod transforms;
pub mod types;

use alloc::string::ToString;
use alloc::{boxed::Box, collections::BTreeMap, format, string::String, vec, vec::Vec};
pub use types::{Expr, MathError};

use num_traits::Float;

pub struct MathCore {
    engine: engine::Engine,
}

impl MathCore {
    pub fn new() -> Self {
        MathCore {
            engine: engine::Engine::new(),
        }
    }

    pub fn parse(expression: &str) -> Result<Expr, MathError> {
        parser::Parser::parse(expression)
    }

    pub fn calculate(&self, expression: &str) -> Result<f64, MathError> {
        let expr = Self::parse(expression)?;
        let result = self.engine.evaluate(&expr)?;

        match result {
            Expr::Number(n) => Ok(n),
            _ => Err(MathError::InvalidOperation(
                "Result is not a real number".to_string(),
            )),
        }
    }

    pub fn evaluate(&self, expression: &str) -> Result<Expr, MathError> {
        let expr = Self::parse(expression)?;
        self.engine.evaluate(&expr)
    }

    pub fn evaluate_with_vars(
        &self,
        expression: &str,
        vars: &BTreeMap<String, f64>,
    ) -> Result<f64, MathError> {
        let expr = Self::parse(expression)?;
        let result = self.engine.evaluate_with_vars(&expr, vars)?;

        match result {
            Expr::Number(n) => Ok(n),
            _ => Err(MathError::InvalidOperation(
                "Result is not a real number".to_string(),
            )),
        }
    }

    pub fn differentiate(expression: &str, var: &str) -> Result<Expr, MathError> {
        let expr = Self::parse(expression)?;
        calculus::Calculus::differentiate(&expr, var)
    }

    pub fn integrate(expression: &str, var: &str) -> Result<Expr, MathError> {
        let expr = Self::parse(expression)?;
        calculus::Calculus::integrate(&expr, var)
    }

    pub fn numerical_integrate(
        expression: &str,
        var: &str,
        lower: f64,
        upper: f64,
    ) -> Result<f64, MathError> {
        let expr = Self::parse(expression)?;
        calculus::Calculus::numerical_integrate(&expr, var, lower, upper, 1000)
    }

    pub fn solve(equation: &str, var: &str) -> Result<Vec<Expr>, MathError> {
        let expr = Self::parse(equation)?;
        solver::Solver::solve(&expr, var)
    }

    pub fn factor(expression: &str) -> Result<Expr, MathError> {
        let expr = Self::parse(expression)?;
        solver::Solver::factor(&expr)
    }

    pub fn simplify(expression: &str) -> Result<Expr, MathError> {
        let expr = Self::parse(expression)?;
        Ok(Self::simplify_expr(&expr))
    }

    // TODO: add more simplification rules

    fn simplify_expr(expr: &Expr) -> Expr {
        use types::{BinaryOp, UnaryOp};

        // recursive simplification
        match expr {
            Expr::Binary { op, left, right } => {
                let left = Self::simplify_expr(left);
                let right = Self::simplify_expr(right);

                match (&left, op, &right) {
                    (Expr::Number(l), BinaryOp::Add, Expr::Number(r)) => Expr::Number(l + r),
                    (Expr::Number(l), BinaryOp::Subtract, Expr::Number(r)) => Expr::Number(l - r),
                    (Expr::Number(l), BinaryOp::Multiply, Expr::Number(r)) => Expr::Number(l * r),
                    (Expr::Number(l), BinaryOp::Divide, Expr::Number(r))
                        if r.abs() > f64::EPSILON =>
                    {
                        Expr::Number(l / r)
                    }
                    (Expr::Number(l), BinaryOp::Power, Expr::Number(r)) => Expr::Number(l.powf(*r)),

                    (e, BinaryOp::Add, other) | (other, BinaryOp::Add, e) if e.is_zero() => {
                        other.clone()
                    }
                    (e, BinaryOp::Subtract, other) if other.is_zero() => e.clone(),
                    (e, BinaryOp::Multiply, _) | (_, BinaryOp::Multiply, e) if e.is_zero() => {
                        Expr::zero()
                    }
                    (e, BinaryOp::Multiply, other) | (other, BinaryOp::Multiply, e)
                        if e.is_one() =>
                    {
                        other.clone()
                    }
                    (e, BinaryOp::Divide, other) if other.is_one() => e.clone(),
                    (_e, BinaryOp::Power, other) if other.is_zero() => Expr::one(),
                    (e, BinaryOp::Power, other) if other.is_one() => e.clone(),

                    (Expr::Symbol(s1), BinaryOp::Subtract, Expr::Symbol(s2)) if s1 == s2 => {
                        Expr::zero()
                    }

                    _ => Expr::Binary {
                        op: *op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                }
            }

            Expr::Unary { op, expr: inner } => {
                let inner = Self::simplify_expr(inner);

                match (op, &inner) {
                    (UnaryOp::Negate, Expr::Number(n)) => Expr::Number(-n),
                    (
                        UnaryOp::Negate,
                        Expr::Unary {
                            op: UnaryOp::Negate,
                            expr: e,
                        },
                    ) => *e.clone(),
                    (UnaryOp::Abs, Expr::Number(n)) => Expr::Number(n.abs()),

                    _ => Expr::Unary {
                        op: *op,
                        expr: Box::new(inner),
                    },
                }
            }

            _ => expr.clone(),
        }
    }

    pub fn plot_ascii(
        expression: &str,
        var: &str,
        x_min: f64,
        x_max: f64,
        width: usize,
        height: usize,
    ) -> Result<String, MathError> {
        let expr = Self::parse(expression)?;
        let engine = engine::Engine::new();

        let mut points = Vec::new();
        let step = (x_max - x_min) / width as f64;

        for i in 0..=width {
            let x = x_min + i as f64 * step;
            let mut vars = BTreeMap::new();
            vars.insert(var.to_string(), x);

            if let Ok(Expr::Number(y)) = engine.evaluate_with_vars(&expr, &vars) {
                if y.is_finite() {
                    points.push((x, y));
                }
            }
        }

        if points.is_empty() {
            return Err(MathError::InvalidOperation(
                "No valid points to plot".to_string(),
            ));
        }

        let y_min = points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = points
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut plot = vec![vec![' '; width + 1]; height + 1];

        #[allow(clippy::needless_range_loop)]
        for y in 0..=height {
            plot[y][0] = '|';
        }
        #[allow(clippy::needless_range_loop)]
        for x in 0..=width {
            plot[height][x] = '-';
        }
        plot[height][0] = '+';

        for (x, y) in points {
            let plot_x = ((x - x_min) / (x_max - x_min) * width as f64) as usize;
            let plot_y = height - ((y - y_min) / (y_max - y_min) * height as f64) as usize;

            if plot_x <= width && plot_y <= height {
                plot[plot_y][plot_x] = '*';
            }
        }

        let mut result = String::new();
        result.push_str(&format!(
            "Plot of {} from x={} to x={}\n",
            expression, x_min, x_max
        ));
        result.push_str(&format!("y range: [{:.2}, {:.2}]\n\n", y_min, y_max));

        for row in plot {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }

        Ok(result)
    }
}

impl Default for MathCore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::println;

    #[test]
    fn test_basic_arithmetic() {
        let math = MathCore::new();

        assert_eq!(math.calculate("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(math.calculate("(2 + 3) * 4").unwrap(), 20.0);
        assert_eq!(math.calculate("2^3").unwrap(), 8.0);
        assert_eq!(math.calculate("10 / 2").unwrap(), 5.0);
    }

    #[test]
    fn test_functions() {
        let math = MathCore::new();

        let sin_val = math.calculate("sin(0)").unwrap();
        assert!(sin_val.abs() < 0.0001);

        let cos_val = math.calculate("cos(0)").unwrap();
        assert!((cos_val - 1.0).abs() < 0.0001);

        let exp_val = math.calculate("exp(0)").unwrap();
        assert!((exp_val - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_differentiation() {
        let deriv = MathCore::differentiate("x^2", "x").unwrap();
        println!("d/dx(x^2) = {}", deriv);

        let deriv = MathCore::differentiate("sin(x)", "x").unwrap();
        println!("d/dx(sin(x)) = {}", deriv);

        let deriv = MathCore::differentiate("x^3 + 2*x^2 + x + 1", "x").unwrap();
        println!("d/dx(x^3 + 2*x^2 + x + 1) = {}", deriv);
    }

    #[test]
    fn test_integration() {
        let integral = MathCore::integrate("x", "x").unwrap();
        println!("∫x dx = {}", integral);

        let integral = MathCore::integrate("x^2", "x").unwrap();
        println!("∫x^2 dx = {}", integral);

        let integral = MathCore::integrate("sin(x)", "x").unwrap();
        println!("∫sin(x) dx = {}", integral);
    }

    #[test]
    fn test_solving() {
        let roots = MathCore::solve("x^2 - 4", "x").unwrap();
        assert_eq!(roots.len(), 2);
        println!("Roots of x^2 - 4 = 0: {:?}", roots);

        let roots = MathCore::solve("x^2 + x - 6", "x").unwrap();
        assert_eq!(roots.len(), 2);
        println!("Roots of x^2 + x - 6 = 0: {:?}", roots);
    }

    #[test]
    fn test_variables() {
        let math = MathCore::new();
        let mut vars = BTreeMap::new();
        vars.insert("a".to_string(), 3.0);
        vars.insert("b".to_string(), 4.0);

        let result = math.evaluate_with_vars("a^2 + b^2", &vars).unwrap();
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_simplification() {
        let simplified = MathCore::simplify("x - x").unwrap();
        assert!(simplified.is_zero());

        let simplified = MathCore::simplify("0 * x").unwrap();
        assert!(simplified.is_zero());

        let simplified = MathCore::simplify("1 * x").unwrap();
        println!("1 * x = {}", simplified);
    }

    #[test]
    fn test_equation_form_solve() {
        // 3 = 6/x  →  x = 2
        let roots = MathCore::solve("3 = 6/x", "x").unwrap();
        assert!(!roots.is_empty());
        if let Expr::Number(n) = &roots[0] {
            assert!((*n - 2.0).abs() < 1e-6, "expected x=2, got {}", n);
        }

        // x^2 - 4 = 0  (explicit zero rhs)
        let roots = MathCore::solve("x^2 - 4 = 0", "x").unwrap();
        assert_eq!(roots.len(), 2);

        // x + 1 = 4  →  x = 3
        let roots = MathCore::solve("x + 1 = 4", "x").unwrap();
        assert_eq!(roots.len(), 1);
        if let Expr::Number(n) = &roots[0] {
            assert!((*n - 3.0).abs() < 1e-10, "expected x=3, got {}", n);
        }
    }

    // ── Arithmetic operations ────────────────────────────────────────────────

    #[test]
    fn test_arithmetic_addition() {
        let math = MathCore::new();
        assert_eq!(math.calculate("1 + 2").unwrap(), 3.0);
        assert_eq!(math.calculate("0.5 + 0.5").unwrap(), 1.0);
        assert_eq!(math.calculate("-3 + 5").unwrap(), 2.0);
    }

    #[test]
    fn test_arithmetic_subtraction() {
        let math = MathCore::new();
        assert_eq!(math.calculate("10 - 3").unwrap(), 7.0);
        assert_eq!(math.calculate("0 - 5").unwrap(), -5.0);
        assert_eq!(math.calculate("2.5 - 1.5").unwrap(), 1.0);
    }

    #[test]
    fn test_arithmetic_multiplication() {
        let math = MathCore::new();
        assert_eq!(math.calculate("3 * 4").unwrap(), 12.0);
        assert_eq!(math.calculate("-2 * 6").unwrap(), -12.0);
        assert_eq!(math.calculate("0 * 999").unwrap(), 0.0);
    }

    #[test]
    fn test_arithmetic_division() {
        let math = MathCore::new();
        assert_eq!(math.calculate("10 / 2").unwrap(), 5.0);
        assert_eq!(math.calculate("1 / 4").unwrap(), 0.25);
        assert!(math.calculate("1 / 0").is_err());
    }

    #[test]
    fn test_arithmetic_power() {
        let math = MathCore::new();
        assert_eq!(math.calculate("2^10").unwrap(), 1024.0);
        assert_eq!(math.calculate("4^0.5").unwrap(), 2.0);
        assert_eq!(math.calculate("27^(1/3)").unwrap(), 3.0);
        assert_eq!(math.calculate("5^0").unwrap(), 1.0);
        assert_eq!(math.calculate("1^999").unwrap(), 1.0);
    }

    #[test]
    fn test_arithmetic_modulo() {
        let math = MathCore::new();
        assert_eq!(math.calculate("10 % 3").unwrap(), 1.0);
        assert_eq!(math.calculate("7 % 7").unwrap(), 0.0);
        assert_eq!(math.calculate("9 % 4").unwrap(), 1.0);
    }

    #[test]
    fn test_arithmetic_factorial() {
        let math = MathCore::new();
        assert_eq!(math.calculate("0!").unwrap(), 1.0);
        assert_eq!(math.calculate("1!").unwrap(), 1.0);
        assert_eq!(math.calculate("5!").unwrap(), 120.0);
        assert_eq!(math.calculate("10!").unwrap(), 3628800.0);
    }

    #[test]
    fn test_arithmetic_absolute_value_pipe() {
        let math = MathCore::new();
        assert_eq!(math.calculate("|3|").unwrap(), 3.0);
        assert_eq!(math.calculate("|-5|").unwrap(), 5.0);
        assert_eq!(math.calculate("|0|").unwrap(), 0.0);
        assert!((math.calculate("|-3.7|").unwrap() - 3.7).abs() < 1e-10);
    }

    #[test]
    fn test_operator_precedence() {
        let math = MathCore::new();
        // multiplication before addition
        assert_eq!(math.calculate("2 + 3 * 4").unwrap(), 14.0);
        // parentheses override precedence
        assert_eq!(math.calculate("(2 + 3) * 4").unwrap(), 20.0);
        // power before multiplication
        assert_eq!(math.calculate("2 * 3^2").unwrap(), 18.0);
    }

    // ── Trigonometric functions ──────────────────────────────────────────────

    #[test]
    fn test_trig_sin() {
        let math = MathCore::new();
        assert!((math.calculate("sin(0)").unwrap()).abs() < 1e-10);
        assert!((math.calculate("sin(pi/2)").unwrap() - 1.0).abs() < 1e-10);
        assert!((math.calculate("sin(pi)").unwrap()).abs() < 1e-10);
        assert!((math.calculate("sin(3*pi/2)").unwrap() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trig_cos() {
        let math = MathCore::new();
        assert!((math.calculate("cos(0)").unwrap() - 1.0).abs() < 1e-10);
        assert!((math.calculate("cos(pi/2)").unwrap()).abs() < 1e-10);
        assert!((math.calculate("cos(pi)").unwrap() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trig_tan() {
        let math = MathCore::new();
        assert!((math.calculate("tan(0)").unwrap()).abs() < 1e-10);
        assert!((math.calculate("tan(pi/4)").unwrap() - 1.0).abs() < 1e-10);
        assert!((math.calculate("tan(-pi/4)").unwrap() + 1.0).abs() < 1e-10);
    }

    // sec(x) = 1/cos(x) — no built-in function, expressed as 1/cos(x)
    #[test]
    fn test_trig_sec_via_cos() {
        let math = MathCore::new();
        // sec(0) = 1/cos(0) = 1
        assert!((math.calculate("1/cos(0)").unwrap() - 1.0).abs() < 1e-10);
        // sec(pi/3) = 1/cos(pi/3) = 2
        assert!((math.calculate("1/cos(pi/3)").unwrap() - 2.0).abs() < 1e-10);
        // sec is also the derivative of tan — d/dx tan(x) = sec²(x)
        // Verify numerically: (tan(x+h) - tan(x-h))/(2h) ≈ 1/cos²(x) at x=pi/4
        let deriv_num = (math.calculate("tan(pi/4 + 0.0001)").unwrap()
            - math.calculate("tan(pi/4 - 0.0001)").unwrap())
            / 0.0002;
        let sec_sq = 1.0 / math.calculate("cos(pi/4)").unwrap().powi(2);
        assert!((deriv_num - sec_sq).abs() < 1e-6);
    }

    // ── Exponential & logarithmic functions ─────────────────────────────────

    #[test]
    fn test_exp() {
        let math = MathCore::new();
        assert!((math.calculate("exp(0)").unwrap() - 1.0).abs() < 1e-10);
        assert!((math.calculate("exp(1)").unwrap() - core::f64::consts::E).abs() < 1e-10);
        assert!((math.calculate("exp(2)").unwrap() - core::f64::consts::E.powi(2)).abs() < 1e-10);
        // exp and ln are inverses
        assert!((math.calculate("exp(ln(5))").unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln() {
        let math = MathCore::new();
        assert!((math.calculate("ln(1)").unwrap()).abs() < 1e-10);
        assert!((math.calculate("ln(e)").unwrap() - 1.0).abs() < 1e-10);
        assert!((math.calculate("ln(e^2)").unwrap() - 2.0).abs() < 1e-10);
        // ln of non-positive should not produce a finite real result
        // (returns symbolic or errors — either is acceptable)
        assert!(math.calculate("ln(0)").is_err() || !math.calculate("ln(0)").unwrap().is_finite());
    }

    #[test]
    fn test_log_custom_base() {
        let math = MathCore::new();
        assert!((math.calculate("log(100, 10)").unwrap() - 2.0).abs() < 1e-10);
        assert!((math.calculate("log(8, 2)").unwrap() - 3.0).abs() < 1e-10);
        assert!((math.calculate("log(1, 10)").unwrap()).abs() < 1e-10);
        assert!((math.calculate("log(e, e)").unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let math = MathCore::new();
        assert!((math.calculate("sqrt(4)").unwrap() - 2.0).abs() < 1e-10);
        assert!((math.calculate("sqrt(9)").unwrap() - 3.0).abs() < 1e-10);
        assert!((math.calculate("sqrt(2)").unwrap() - 2f64.sqrt()).abs() < 1e-10);
        assert!((math.calculate("sqrt(0)").unwrap()).abs() < 1e-10);
        // sqrt(x)^2 = x
        assert!((math.calculate("sqrt(7)^2").unwrap() - 7.0).abs() < 1e-10);
    }

    // ── Utility functions ────────────────────────────────────────────────────

    #[test]
    fn test_min() {
        let math = MathCore::new();
        assert_eq!(math.calculate("min(3, 1, 2)").unwrap(), 1.0);
        assert_eq!(math.calculate("min(5, 5)").unwrap(), 5.0);
        assert_eq!(math.calculate("min(-1, -2, -3)").unwrap(), -3.0);
        assert_eq!(math.calculate("min(0, 1)").unwrap(), 0.0);
    }

    #[test]
    fn test_max() {
        let math = MathCore::new();
        assert_eq!(math.calculate("max(3, 1, 2)").unwrap(), 3.0);
        assert_eq!(math.calculate("max(5, 5)").unwrap(), 5.0);
        assert_eq!(math.calculate("max(-1, -2, -3)").unwrap(), -1.0);
        assert_eq!(math.calculate("max(0, 1)").unwrap(), 1.0);
    }

    #[test]
    fn test_abs_function() {
        let math = MathCore::new();
        assert_eq!(math.calculate("abs(3)").unwrap(), 3.0);
        assert_eq!(math.calculate("abs(-5)").unwrap(), 5.0);
        assert_eq!(math.calculate("abs(0)").unwrap(), 0.0);
        // abs and pipe-syntax should agree
        assert_eq!(
            math.calculate("abs(-7)").unwrap(),
            math.calculate("|-7|").unwrap()
        );
    }

    // ── Mathematical constants ───────────────────────────────────────────────

    #[test]
    fn test_constant_pi() {
        let math = MathCore::new();
        let pi = math.calculate("pi").unwrap();
        assert!((pi - core::f64::consts::PI).abs() < 1e-15);
        // pi appears correctly inside expressions
        assert!((math.calculate("2*pi").unwrap() - core::f64::consts::TAU).abs() < 1e-15);
    }

    #[test]
    fn test_constant_e() {
        let math = MathCore::new();
        let e = math.calculate("e").unwrap();
        assert!((e - core::f64::consts::E).abs() < 1e-15);
        // e^1 = e
        assert!((math.calculate("e^1").unwrap() - core::f64::consts::E).abs() < 1e-15);
    }

    #[test]
    fn test_constant_tau() {
        let math = MathCore::new();
        let tau = math.calculate("tau").unwrap();
        assert!((tau - core::f64::consts::TAU).abs() < 1e-15);
        // tau = 2*pi
        assert!((math.calculate("tau - 2*pi").unwrap()).abs() < 1e-14);
    }
}
