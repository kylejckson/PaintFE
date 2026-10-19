pub(crate) fn evaluate_dimension_expression(input: &str) -> Option<f32> {
    #[derive(Clone, Copy)]
    struct Parser<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn new(src: &'a str) -> Self {
            Self {
                bytes: src.as_bytes(),
                pos: 0,
            }
        }

        fn skip_ws(&mut self) {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
        }

        fn parse_expr(&mut self) -> Option<f32> {
            let mut value = self.parse_term()?;
            loop {
                self.skip_ws();
                let Some(&op) = self.bytes.get(self.pos) else {
                    break;
                };
                if op != b'+' && op != b'-' {
                    break;
                }
                self.pos += 1;
                let rhs = self.parse_term()?;
                value = if op == b'+' { value + rhs } else { value - rhs };
            }
            Some(value)
        }

        fn parse_term(&mut self) -> Option<f32> {
            let mut value = self.parse_factor()?;
            loop {
                self.skip_ws();
                let Some(&op) = self.bytes.get(self.pos) else {
                    break;
                };
                if op != b'*' && op != b'/' {
                    break;
                }
                self.pos += 1;
                let rhs = self.parse_factor()?;
                value = if op == b'*' {
                    value * rhs
                } else {
                    if rhs.abs() < f32::EPSILON {
                        return None;
                    }
                    value / rhs
                };
            }
            Some(value)
        }

        fn parse_factor(&mut self) -> Option<f32> {
            self.skip_ws();
            if let Some(&b'+') = self.bytes.get(self.pos) {
                self.pos += 1;
                return self.parse_factor();
            }
            if let Some(&b'-') = self.bytes.get(self.pos) {
                self.pos += 1;
                return self.parse_factor().map(|v| -v);
            }
            if let Some(&b'(') = self.bytes.get(self.pos) {
                self.pos += 1;
                let value = self.parse_expr()?;
                self.skip_ws();
                if self.bytes.get(self.pos) == Some(&b')') {
                    self.pos += 1;
                    return Some(value);
                }
                return None;
            }
            self.parse_number()
        }

        fn parse_number(&mut self) -> Option<f32> {
            self.skip_ws();
            let start = self.pos;
            let mut seen_digit = false;
            let mut seen_dot = false;
            while let Some(&ch) = self.bytes.get(self.pos) {
                if ch.is_ascii_digit() {
                    seen_digit = true;
                    self.pos += 1;
                } else if ch == b'.' && !seen_dot {
                    seen_dot = true;
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if !seen_digit {
                return None;
            }
            std::str::from_utf8(&self.bytes[start..self.pos])
                .ok()?
                .parse::<f32>()
                .ok()
        }
    }

    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut parser = Parser::new(trimmed);
    let value = parser.parse_expr()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() || !value.is_finite() {
        return None;
    }
    Some(value)
}
