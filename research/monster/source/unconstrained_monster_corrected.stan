// unconstrained_monster.stan (this directory) with the two bugs corrected:
//   - lean-body fraction of the non-fat organ volumes 0.873, not 0.837
//     (Bois et al. 1996, Arch Toxicol 70:350; Gelman, Bois and Jiang 1996,
//     JASA eq. 1);
//   - alveolar air (VPR * C_inh + C_ven) / (VPR + Pba), not
//     (C_inh + C_ven) / (VPR + Pba) (the lung mass balance of MCSim's
//     perc.model).
// Apart from this header, the diff against unconstrained_monster.stan is these
// three lines (both `.837`s and the alveolar line).
// The Monster model as `stan/unconstrained_monster.stan` in
// https://github.com/nsiccha/monster at commit
// a362738efd9c83525a2edb6db3e3646fba1a293f, unchanged except for the syntax
// that current stanc (2.33+) no longer accepts:
//   - old array declarations (`real x[n]`, `real[,] f(...)`) are written as
//     `array[n] real x` / `array[,] real f(...)`;
//   - `#` comments are written as `//`;
//   - real-valued conditions (`if(likelihood)`, `if(!condition...)`,
//     `if(weight)`) compare against zero explicitly.
// Its data is `cfg/nu=4/parallel_incremental_data.json` from the same commit,
// copied verbatim as `parallel_incremental_data.json`.
functions {
  vector dydt_exposure(
    real t, vector concentration_out,
    vector FVP, vector FFPF, real CFFPF,
    real VMI, real KMI
  ){
    return FVP .* (
        dot_product(FFPF, concentration_out) + CFFPF - concentration_out
    ) + [
        0,0,0,VMI * concentration_out[4] / (KMI + concentration_out[4])
    ]';
  }
  vector dydt_nonexposure(
    real t, vector concentration_out,
    vector FVP, vector FFPF,
    real VMI, real KMI
  ){
    return FVP .* (
        dot_product(FFPF, concentration_out) - concentration_out
    ) + [
        0,0,0,VMI * concentration_out[4] / (KMI + concentration_out[4])
    ]';
  }
  real lambert_w0_exp(real earg){
    if(is_nan(earg)){return earg;}
    // https://www.wolframalpha.com/input/?i=N%5BTable%5B%7BProductLog%5BExp%5Bx%5D%5D%2CProductLog%5BExp%5Bx%5D%5D%2F%28ProductLog%5BExp%5Bx%5D%5D%2B1%29%7D%2C+%7Bx%2C%7B-40%2C700%7D%7D%5D%2C+32%5D
    // (4.2483542552915889772807209044045×10^-18 | 4.2483542552915889592322070259504×10^-18
    // 693.45830887902549833674969928122 | 0.99856002874871758528483054395180)
    if(earg > 700){
      real x0 = 700;
      real y0 = lambert_w0(exp(x0));
      real dydx = y0/(y0+1);
      return y0 + (earg - x0) * dydx;
      // real y0 = 693.45830887902549833674969928122;
      // real dydx = 0.9985600287487175852848305439518009385629359235843258625030446976;
      // // https://www.wolframalpha.com/input/?i=ProductLog%5BExp%5B700%5D%5D%2F%28ProductLog%5BExp%5B700%5D%5D%2B1%29
      // return y0 + (earg - x0) * dydx;
    }
    if(earg < -40){
      real x0 = -40;
      real y0 = lambert_w0(exp(x0));
      // real y0 = 4.2483542552915889772807209044045064097730030864961656941893e-18;
      return y0 * exp(earg - x0);
    }
    return lambert_w0(exp(earg));
  }
  real min_concentration(){return 1e-12;}
  real exact_michaelis_menten_solution(real dt, real C, real V, real K){
    if(C <= min_concentration()){return min_concentration();}
    if(K == 0){return C - dt * V;}
    real earg = (dt*V+C)/K+log(C/K);
    return K * lambert_w0_exp(earg);
  }
  vector ginterpolate(real xi, vector left, vector right){
    return exp((1-xi)*log(min_concentration()+left)+xi*log(min_concentration()+right));
  }
  // vector strang_step(real dt)
  array[,] real simulate_person(
      data real concentration_exposure,
      data array[] real times,
      vector params,
      data vector measured_params,
      data int no_sub_steps
  ){
    int no_times = size(times);
    int no_exposure_times = 1;
    array[no_times, 2] real rv;
    real lean_body_mass = measured_params[1]; //[kg]
    real mass_fraction_fat = measured_params[2]; //[1]
    real volume_flow_pulmonary = measured_params[3]; //[l/min]
    // real concentration_exposure = measured_params[4]; //[mug/l]

    real body_mass = lean_body_mass / (1 - mass_fraction_fat); //[kg]
    real volume_fat = mass_fraction_fat * body_mass / .92; //[l]
    real volume_flow_alveolar = .7 * volume_flow_pulmonary; //[l/min]
    real mass_flow_exposure = volume_flow_alveolar * concentration_exposure; //[mug/min]

    real VPR = params[1]; //[1]
    real volume_flow_venous = volume_flow_alveolar / VPR; //[l/min]
    vector[4] unit_volume_flow = params[2:5];
    vector[4] volume_flow = unit_volume_flow * volume_flow_venous; //[l/min]
    // body density = water density = 1kg/l
    vector[3] dummy = lean_body_mass * params[6:8]; //[l]
    vector[4] volume = append_row( //[l]
      dummy[1:2],
      [volume_fat, dummy[3]]'
    );
    real partition_coefficient_alveolar = params[9]; //[1]
    vector[4] partition_coefficient = params[10:13]; //[1]
    vector[4] effective_volume = volume .* partition_coefficient;
    real VMI = pow(lean_body_mass, .7) * params[14]/(effective_volume[4]);
    real KMI = params[15]/effective_volume[4];

    vector[4] concentration_out = rep_vector(min_concentration(), 4); //[mug/l]
    vector[4] FVP = volume_flow ./ (effective_volume); //[1/min]
    real FPF = volume_flow_venous + volume_flow_alveolar / partition_coefficient_alveolar; //[l/min]

    array[no_times] vector[4] all_concentration_out;
    if(no_sub_steps <= 0){
      all_concentration_out[1] = ode_bdf_tol(
        dydt_exposure, concentration_out, 0, {times[1]},
        pow(10, no_sub_steps), pow(10, no_sub_steps-14), 100000000,
        FVP, volume_flow / FPF, mass_flow_exposure / FPF,
        -VMI, KMI
      )[1];
      all_concentration_out[2:] = ode_bdf_tol(
        dydt_nonexposure, all_concentration_out[1], 0, to_array_1d(
          to_vector(times[2:])-times[1]
        ),
        pow(10, no_sub_steps), pow(10, no_sub_steps-14), 100000000,
        FVP, volume_flow / FPF,
        -VMI, KMI
      );
    }else{
      matrix[4,4] A = add_diag( //[1/min]
        FVP * (volume_flow / FPF)',
        -FVP
      );
      vector[4] A_source = mass_flow_exposure / FPF * A \ FVP; //[mug/l]
      vector[4] last_concentration_out = concentration_out; //[mug/l]
      real dt = times[1] / no_sub_steps; //[min]
      matrix[4,4] transition_matrix = matrix_exp(dt * A); //[1]
      vector[4] exp_A_source = transition_matrix * A_source - A_source; //[mug/l]
      real last_time = 0;  //[min]
      real next_time = 0;
      int time_idx = 1;
      real next_checkpoint = times[time_idx];
      while(time_idx <= no_times){
        next_time = last_time + dt;
        concentration_out[4] = exact_michaelis_menten_solution(
          dt/2, concentration_out[4], -VMI, KMI
        );
        if(time_idx <= no_exposure_times){
          concentration_out = transition_matrix * concentration_out + exp_A_source;
        }else{
          concentration_out = transition_matrix * concentration_out;
        }
        concentration_out[4] = exact_michaelis_menten_solution(
          dt/2, concentration_out[4], -VMI, KMI
        );
        while(next_time >= next_checkpoint){
          all_concentration_out[time_idx] = ginterpolate(
            (next_checkpoint - last_time)/dt,
            last_concentration_out,
            concentration_out
          );
          if(time_idx == no_exposure_times){
            concentration_out = all_concentration_out[time_idx];
            next_time = times[time_idx];
          }
          time_idx += 1;
          if(time_idx <= no_times){
            next_checkpoint = times[time_idx];
          }else{
            break;
          }
        }
        last_time = next_time;
        last_concentration_out = concentration_out;
      }
    }
    for(time_idx in 1:no_times){
      concentration_out = all_concentration_out[time_idx];
      real concentration_venous = dot_product(unit_volume_flow, concentration_out);
      real concentration_inhale = (time_idx <= no_exposure_times ? concentration_exposure : 0);
      real concentration_alveolar = (
        VPR * concentration_inhale + concentration_venous
      ) / (
        VPR + partition_coefficient_alveolar
      );
      real concentration_exhale = .7 * concentration_alveolar + .3 * concentration_inhale;
      rv[time_idx] = {min_concentration()+concentration_venous, min_concentration()+concentration_exhale};
    }
    return rv;
  }
}
data {
  int no_persons;
  int no_measured_params;
  array[no_persons] vector[no_measured_params] measured_params;
  int no_experiments;
  array[no_experiments] real exposures;
  int no_measurements;
  array[no_persons, no_experiments, no_measurements, 3] real experiments;
  array[no_persons, no_experiments, no_measurements, 2] real weights;
  int no_latent_params;
  vector<lower=0>[no_latent_params] population_eM_eM;
  vector<lower=0>[no_latent_params] population_eM_eS;
  vector<lower=1>[no_latent_params] population_eS_mu;
  vector<lower=0>[no_latent_params] population_eS_nu;
  array[2] vector[no_latent_params] population_truncation;
  array[2] vector[no_latent_params] std_truncation;
  array[2] vector[no_latent_params] person_truncation;
  real noise_scale;

  real likelihood;
  int no_sub_steps;
  int no_sim_sub_steps;
}
transformed data {
  vector[no_latent_params] log_population_eM_eS = log(population_eM_eS);
  vector<lower=0>[no_latent_params] log_population_eS_mu = log(population_eS_mu);
  vector[no_latent_params] log_log_population_eS_mu = log(log_population_eS_mu);
  array[no_persons, no_experiments] real condition_on_experiment;
  for(person in 1:no_persons){
    for(experiment in 1:no_experiments){
      condition_on_experiment[person,experiment] = max(to_array_1d(weights[person,experiment]));
    }
  }
}
parameters {
  vector<
    lower=population_truncation[1],
    upper=population_truncation[2]
  >[no_latent_params] unit_log_population_eM;
  vector<
    lower=std_truncation[1],
    upper=std_truncation[2]
  >[no_latent_params] unit_log_population_eS;
  // array[no_persons] vector<
  //   lower=rep_array(
  //     (
  //       (person_truncation[1] - unit_log_population_eM) .* log_population_eM_eS
  //     ) ./ exp(
  //       log_log_population_eS_mu + unit_log_population_eS
  //     ),
  //     no_persons
  //   ),
  //   upper=rep_array(
  //     (
  //       (person_truncation[2] - unit_log_population_eM) .* log_population_eM_eS
  //     ) ./ exp(
  //       log_log_population_eS_mu + unit_log_population_eS
  //     ),
  //     no_persons
  //   )
  // >[no_latent_params] unit_log_person_params;
  array[no_persons] vector[no_latent_params] unit_log_person_params;
  vector<lower=0>[2] noise;
}
transformed parameters {
  vector<lower=0>[no_latent_params] population_eM = (
      population_eM_eM .* pow(population_eM_eS, unit_log_population_eM)
  );
  vector<lower=0>[no_latent_params] log_population_eS = exp(
      log_log_population_eS_mu + unit_log_population_eS
  );
  vector<lower=1>[no_latent_params] population_eS = exp(
    log_population_eS
  );
  array[no_persons] vector<lower=0>[no_latent_params] person_params;
  array[no_persons, no_experiments, no_measurements, 2] real true_states;
  population_eM[2:5] = population_eM[2:5] / sum(population_eM[2:5]);
  population_eM[6:7] = (
    .873 - population_eM[8]
  ) * population_eM[6:7] / sum(population_eM[6:7]);
  for(person in 1:no_persons){
    person_params[person] = (
      population_eM .* pow(population_eS, unit_log_person_params[person])
    );
    person_params[person, 2:5] = person_params[person, 2:5] / sum(person_params[person, 2:5]);
    person_params[person,6:7] = (
      .873 - person_params[person,8]
    ) * person_params[person,6:7] / sum(person_params[person,6:7]);
    if(likelihood != 0){
      for(experiment in 1:no_experiments){
        if(condition_on_experiment[person, experiment] == 0){
          continue;
        }
        true_states[person, experiment] = simulate_person(
          exposures[experiment],
          experiments[person, experiment,,1],
          person_params[person], measured_params[person],
          no_sub_steps
        );
      }
    }
  }
}
model {
  for(person in 1:no_persons){
    target += normal_lpdf(unit_log_person_params[person] | 0,1);
  }
  target += normal_lpdf(unit_log_population_eM | 0,1);
  target += scaled_inv_chi_square_lpdf(
    pow(log_population_eS, 2) | population_eS_nu, log_population_eS_mu
  );
  target += log(pow(log_population_eS,2));
  if(likelihood != 0){
    target += -log(noise);
  }else{
    target += weibull_lpdf(noise/noise_scale | 2,1);
  }
  if(likelihood != 0){
    for(person in 1:no_persons){
      for(experiment in 1:no_experiments){
        if(condition_on_experiment[person, experiment] == 0){
          continue;
        }
        for(measurement in 1:no_measurements){
          for(i in 1:2){
            real weight = weights[person, experiment, measurement, i];
            if(weight != 0){
              real observation = experiments[person, experiment, measurement, 1+i];
              real state = true_states[person, experiment, measurement, i];
              target += weight * lognormal_lpdf(
                observation | log(state), noise[i]
              );
            }
          }
        }
      }
    }
  }
}
generated quantities {
  array[no_persons, no_experiments, no_measurements, 2] real predicted_states;
  for(person in 1:no_persons){
    for(experiment in 1:no_experiments){
      array[no_measurements, 2] real temp = simulate_person(
        exposures[experiment],
        experiments[person, experiment,,1],
        person_params[person], measured_params[person],
        no_sim_sub_steps
      );
      for(i in 1:2){
        predicted_states[person,experiment,,i] = lognormal_rng(
          log(temp[,i]), noise[i]
        );
      }
    }
  }
}
