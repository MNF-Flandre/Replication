use std::slice;

const ERR_NULL_POINTER: i32 = -1;
const ERR_BAD_ARGS: i32 = -2;
const ERR_LABEL_RANGE: i32 = -3;

const MIN_PROB: f64 = 1e-300;

#[unsafe(no_mangle)]
pub extern "C" fn rolling_max_drawdown_grouped(
    n: usize,
    returns_ptr: *const f64,
    group_ids_ptr: *const i64,
    window: usize,
    min_periods: usize,
    out_ptr: *mut f64,
) -> i32 {
    if returns_ptr.is_null() || group_ids_ptr.is_null() || out_ptr.is_null() {
        return ERR_NULL_POINTER;
    }
    if n == 0 || window == 0 || min_periods == 0 {
        return ERR_BAD_ARGS;
    }

    let returns = unsafe { slice::from_raw_parts(returns_ptr, n) };
    let group_ids = unsafe { slice::from_raw_parts(group_ids_ptr, n) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, n) };

    let mut group_start = 0usize;
    for i in 0..n {
        if i > 0 && group_ids[i] != group_ids[i - 1] {
            group_start = i;
        }
        let start = group_start.max(i.saturating_add(1).saturating_sub(window));
        out[i] = rolling_window_max_drawdown(&returns[start..=i], min_periods);
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn fit_diagonal_gaussian_hmm_seeded(
    n: usize,
    n_states: usize,
    n_features: usize,
    x_ptr: *const f64,
    labels_ptr: *const i64,
    max_iter: usize,
    tol: f64,
    start_prior: f64,
    trans_prior: f64,
    covariance_floor: f64,
    start_out_ptr: *mut f64,
    trans_out_ptr: *mut f64,
    means_out_ptr: *mut f64,
    covars_out_ptr: *mut f64,
    monitor_out_ptr: *mut f64,
    iterations_out_ptr: *mut usize,
) -> i32 {
    if x_ptr.is_null()
        || labels_ptr.is_null()
        || start_out_ptr.is_null()
        || trans_out_ptr.is_null()
        || means_out_ptr.is_null()
        || covars_out_ptr.is_null()
        || monitor_out_ptr.is_null()
        || iterations_out_ptr.is_null()
    {
        return ERR_NULL_POINTER;
    }
    if n == 0 || n_states == 0 || n_features == 0 || max_iter == 0 {
        return ERR_BAD_ARGS;
    }

    let x = unsafe { slice::from_raw_parts(x_ptr, n * n_features) };
    let labels = unsafe { slice::from_raw_parts(labels_ptr, n) };
    let start_out = unsafe { slice::from_raw_parts_mut(start_out_ptr, n_states) };
    let trans_out = unsafe { slice::from_raw_parts_mut(trans_out_ptr, n_states * n_states) };
    let means_out = unsafe { slice::from_raw_parts_mut(means_out_ptr, n_states * n_features) };
    let covars_out = unsafe { slice::from_raw_parts_mut(covars_out_ptr, n_states * n_features) };
    let monitor_out = unsafe { slice::from_raw_parts_mut(monitor_out_ptr, max_iter) };
    let iterations_out = unsafe { &mut *iterations_out_ptr };

    let mut start = vec![0.0f64; n_states];
    let mut trans = vec![trans_prior; n_states * n_states];
    let mut means = vec![0.0f64; n_states * n_features];
    let mut covars = vec![0.0f64; n_states * n_features];

    let init_status = initialize_seeded(
        x,
        labels,
        n,
        n_states,
        n_features,
        start_prior,
        covariance_floor,
        &mut start,
        &mut trans,
        &mut means,
        &mut covars,
    );
    if init_status != 0 {
        return init_status;
    }
    normalize_row_major(&mut trans, n_states, n_states);

    let mut b = vec![0.0f64; n * n_states];
    let mut alpha = vec![0.0f64; n * n_states];
    let mut beta = vec![0.0f64; n * n_states];
    let mut scale = vec![0.0f64; n];
    let mut gamma = vec![0.0f64; n * n_states];
    let mut xi_sum = vec![0.0f64; n_states * n_states];
    let mut new_means = vec![0.0f64; n_states * n_features];
    let mut new_covars = vec![0.0f64; n_states * n_features];
    let mut weights = vec![0.0f64; n_states];

    let mut last_loglik = f64::NEG_INFINITY;
    *iterations_out = 0;

    for iter in 0..max_iter {
        emission_prob(x, n, n_states, n_features, &means, &covars, &mut b);
        let loglik = forward_backward(
            n, n_states, &start, &trans, &b, &mut alpha, &mut beta, &mut scale, &mut gamma,
        );
        monitor_out[iter] = loglik;

        xi_sum.fill(trans_prior);
        for t in 0..n.saturating_sub(1) {
            let mut denom = 0.0f64;
            for i in 0..n_states {
                for j in 0..n_states {
                    denom += alpha[t * n_states + i]
                        * trans[i * n_states + j]
                        * b[(t + 1) * n_states + j]
                        * beta[(t + 1) * n_states + j];
                }
            }
            if denom > 0.0 {
                for i in 0..n_states {
                    for j in 0..n_states {
                        let numer = alpha[t * n_states + i]
                            * trans[i * n_states + j]
                            * b[(t + 1) * n_states + j]
                            * beta[(t + 1) * n_states + j];
                        xi_sum[i * n_states + j] += numer / denom;
                    }
                }
            }
        }

        weights.fill(1e-12);
        new_means.fill(0.0);
        for t in 0..n {
            for state in 0..n_states {
                let g = gamma[t * n_states + state];
                weights[state] += g;
                for feature in 0..n_features {
                    new_means[state * n_features + feature] += g * x[t * n_features + feature];
                }
            }
        }
        for state in 0..n_states {
            for feature in 0..n_features {
                new_means[state * n_features + feature] /= weights[state];
            }
        }

        new_covars.fill(0.0);
        for t in 0..n {
            for state in 0..n_states {
                let g = gamma[t * n_states + state];
                for feature in 0..n_features {
                    let diff =
                        x[t * n_features + feature] - new_means[state * n_features + feature];
                    new_covars[state * n_features + feature] += g * diff * diff;
                }
            }
        }
        for state in 0..n_states {
            for feature in 0..n_features {
                let value = new_covars[state * n_features + feature] / weights[state];
                new_covars[state * n_features + feature] = value.max(covariance_floor);
            }
        }

        for state in 0..n_states {
            start[state] = gamma[state] + start_prior;
        }
        normalize_vector(&mut start);
        normalize_row_major(&mut xi_sum, n_states, n_states);

        trans.copy_from_slice(&xi_sum);
        means.copy_from_slice(&new_means);
        covars.copy_from_slice(&new_covars);
        *iterations_out = iter + 1;

        if last_loglik.is_finite() && (loglik - last_loglik).abs() < tol {
            break;
        }
        last_loglik = loglik;
    }

    start_out.copy_from_slice(&start);
    trans_out.copy_from_slice(&trans);
    means_out.copy_from_slice(&means);
    covars_out.copy_from_slice(&covars);
    0
}

fn rolling_window_max_drawdown(values: &[f64], min_periods: usize) -> f64 {
    let mut finite_count = 0usize;
    for &value in values {
        if value.is_finite() {
            finite_count += 1;
        }
    }
    if finite_count < min_periods || finite_count <= 1 {
        return f64::NAN;
    }

    let mut wealth = 1.0f64;
    let mut peak = f64::NAN;
    let mut min_drawdown = 0.0f64;
    for &ret in values {
        if !ret.is_finite() {
            continue;
        }
        wealth *= 1.0 + ret;
        if peak.is_nan() || wealth > peak {
            peak = wealth;
        }
        if peak.is_finite() && peak != 0.0 {
            let drawdown = wealth / peak - 1.0;
            if drawdown < min_drawdown {
                min_drawdown = drawdown;
            }
        }
    }
    -min_drawdown
}

fn initialize_seeded(
    x: &[f64],
    labels: &[i64],
    n: usize,
    n_states: usize,
    n_features: usize,
    start_prior: f64,
    covariance_floor: f64,
    start: &mut [f64],
    trans: &mut [f64],
    means: &mut [f64],
    covars: &mut [f64],
) -> i32 {
    for &label in labels {
        if label < 0 || label as usize >= n_states {
            return ERR_LABEL_RANGE;
        }
    }

    start.fill(start_prior);
    start[labels[0] as usize] += 1.0;
    normalize_vector(start);

    for t in 0..n.saturating_sub(1) {
        let prev = labels[t] as usize;
        let cur = labels[t + 1] as usize;
        trans[prev * n_states + cur] += 1.0;
    }

    let mut counts = vec![0usize; n_states];
    let mut global_mean = vec![0.0f64; n_features];
    let mut global_var = vec![0.0f64; n_features];
    for t in 0..n {
        for feature in 0..n_features {
            global_mean[feature] += x[t * n_features + feature];
        }
    }
    for feature in 0..n_features {
        global_mean[feature] /= n as f64;
    }
    for t in 0..n {
        for feature in 0..n_features {
            let diff = x[t * n_features + feature] - global_mean[feature];
            global_var[feature] += diff * diff;
        }
    }
    for feature in 0..n_features {
        global_var[feature] = global_var[feature] / n as f64 + covariance_floor;
    }

    means.fill(0.0);
    for t in 0..n {
        let state = labels[t] as usize;
        counts[state] += 1;
        for feature in 0..n_features {
            means[state * n_features + feature] += x[t * n_features + feature];
        }
    }
    for state in 0..n_states {
        if counts[state] > 0 {
            for feature in 0..n_features {
                means[state * n_features + feature] /= counts[state] as f64;
            }
        } else {
            for feature in 0..n_features {
                means[state * n_features + feature] = global_mean[feature];
            }
        }
    }

    covars.fill(0.0);
    for t in 0..n {
        let state = labels[t] as usize;
        for feature in 0..n_features {
            let diff = x[t * n_features + feature] - means[state * n_features + feature];
            covars[state * n_features + feature] += diff * diff;
        }
    }
    for state in 0..n_states {
        for feature in 0..n_features {
            covars[state * n_features + feature] = if counts[state] > 0 {
                covars[state * n_features + feature] / counts[state] as f64 + covariance_floor
            } else {
                global_var[feature]
            }
            .max(covariance_floor);
        }
    }
    0
}

fn emission_prob(
    x: &[f64],
    n: usize,
    n_states: usize,
    n_features: usize,
    means: &[f64],
    covars: &[f64],
    out: &mut [f64],
) {
    let constant = n_features as f64 * (2.0 * std::f64::consts::PI).ln();
    for t in 0..n {
        let row_start = t * n_states;
        let mut max_log = f64::NEG_INFINITY;
        for state in 0..n_states {
            let mut log_det = 0.0f64;
            let mut quad = 0.0f64;
            for feature in 0..n_features {
                let covar = covars[state * n_features + feature];
                let diff = x[t * n_features + feature] - means[state * n_features + feature];
                log_det += covar.ln();
                quad += diff * diff / covar;
            }
            let log_prob = -0.5 * (constant + log_det + quad);
            out[row_start + state] = log_prob;
            if log_prob > max_log {
                max_log = log_prob;
            }
        }
        for state in 0..n_states {
            out[row_start + state] = (out[row_start + state] - max_log).exp() + MIN_PROB;
        }
    }
}

fn forward_backward(
    n: usize,
    n_states: usize,
    start: &[f64],
    trans: &[f64],
    b: &[f64],
    alpha: &mut [f64],
    beta: &mut [f64],
    scale: &mut [f64],
    gamma: &mut [f64],
) -> f64 {
    for state in 0..n_states {
        alpha[state] = start[state] * b[state];
    }
    scale[0] = row_sum(alpha, 0, n_states);
    let denom0 = scale[0].max(MIN_PROB);
    for state in 0..n_states {
        alpha[state] /= denom0;
    }

    for t in 1..n {
        for state in 0..n_states {
            let mut value = 0.0f64;
            for prev in 0..n_states {
                value += alpha[(t - 1) * n_states + prev] * trans[prev * n_states + state];
            }
            alpha[t * n_states + state] = value * b[t * n_states + state];
        }
        scale[t] = row_sum(alpha, t, n_states);
        let denom = scale[t].max(MIN_PROB);
        for state in 0..n_states {
            alpha[t * n_states + state] /= denom;
        }
    }

    let last = (n - 1) * n_states;
    for state in 0..n_states {
        beta[last + state] = 1.0;
    }
    if n > 1 {
        for t in (0..n - 1).rev() {
            for state in 0..n_states {
                let mut value = 0.0f64;
                for next in 0..n_states {
                    value += trans[state * n_states + next]
                        * b[(t + 1) * n_states + next]
                        * beta[(t + 1) * n_states + next];
                }
                beta[t * n_states + state] = value / scale[t + 1].max(MIN_PROB);
            }
        }
    }

    for t in 0..n {
        let mut denom = 0.0f64;
        for state in 0..n_states {
            let value = alpha[t * n_states + state] * beta[t * n_states + state];
            gamma[t * n_states + state] = value;
            denom += value;
        }
        let denom = denom.max(MIN_PROB);
        for state in 0..n_states {
            gamma[t * n_states + state] /= denom;
        }
    }

    scale.iter().map(|value| value.max(MIN_PROB).ln()).sum()
}

fn normalize_vector(values: &mut [f64]) {
    let mut total: f64 = values.iter().sum();
    if total <= 0.0 {
        total = 1.0;
    }
    for value in values {
        *value /= total;
    }
}

fn normalize_row_major(values: &mut [f64], rows: usize, cols: usize) {
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        normalize_vector(&mut values[start..end]);
    }
}

fn row_sum(values: &[f64], row: usize, cols: usize) -> f64 {
    let start = row * cols;
    values[start..start + cols].iter().sum()
}

#[cfg(test)]
mod tests {
    use super::rolling_window_max_drawdown;

    #[test]
    fn computes_basic_drawdown() {
        let values = [0.1, -0.2, 0.05, -0.1];
        let got = rolling_window_max_drawdown(&values, 2);
        assert!((got - 0.244).abs() < 1e-12);
    }

    #[test]
    fn ignores_nan_for_min_periods_and_path() {
        let values = [f64::NAN, 0.1, -0.2, f64::NAN, 0.05];
        let got = rolling_window_max_drawdown(&values, 2);
        assert!((got - 0.2).abs() < 1e-12);
    }

    #[test]
    fn returns_nan_when_too_few_finite_values() {
        let values = [f64::NAN, 0.1, f64::NAN];
        assert!(rolling_window_max_drawdown(&values, 2).is_nan());
    }

    #[test]
    fn matches_numpy_peak_start_for_negative_first_return() {
        let values = [-0.1, -0.1, 0.05];
        let got = rolling_window_max_drawdown(&values, 2);
        assert!((got - 0.1).abs() < 1e-12);
    }
}
