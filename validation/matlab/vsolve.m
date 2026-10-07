% vsolve.m - result export of one MATLAB-as-shipped solve (validation doc section 7, tag mat_ship).
%
% Driver of validation/matlab_batch.py (mode solve), run in the BASE workspace right after MLTP_solve.m (MLTP.m cut
% behind data.t_opt, the 7-state init captured in vsolve_init.mat, IPOPT max_wall_time cap added):
%   matlab -batch "fclose(fopen('vs_up','w')); MLTP_solve; vsolve"
% MLTP.m clears the workspace, so the paths come from vsolve_cfg.json (written by matlab_batch.py):
%   circuit, mat_file, json_file.
% All own variables are prefixed vs_ so nothing of the model workspace is touched.
%
% mat_file (MAT v7; the names of the Python results where they exist):
%   w_opt lam_g lam_x f            full(sol.x), full(sol.lam_g), full(sol.lam_x), full(sol.f); w is scaled, order MLTP.m:371
%   x_opt u_opt xc_opt t_opt lap   physical units (data.*), lap = t_opt(end)
%   s_knot k_knot N OPT_ds n_w n_g
%   stats          23-state IPOPT: iter_count return_status success t_wall_total t_proc_total, every t_wall_nlp_* and n_call_nlp_*
%   iterations     its trace (inf_pr inf_du obj mu ...)
%   elapsedTime    [0 transcription solver_call], labels in elapsedTime_labels: MLTP.m:387 (vehModel.m + NLP build; the
%                  MLTP.m timer starts at :33, after the 7-state init) and :394 (the solver call)
%   init           7-state init: stats (as above), elapsedTime [0 transcription solver_call] (MLTP_initial.m:238, :245), lap
%   ipopt          options as used: shipped + the runner's max_wall_time (+ max_iter in a smoke run)
%   peak_ws_bytes  PeakWorkingSet64 of this MATLAB process (the whole run so far), NaN if .NET is unavailable
%   matlab_version casadi_version provenance_ok (every script came from this copy, not from the MATLAB path)
% json_file: the scalars of the same (+ circuit, inf_pr and inf_du of the last iteration); matlab_batch.py then adds
% wall, startup, rc, timed_out, git sha and the edited lines.
import casadi.*

vs_cfg = jsondecode(fileread('vsolve_cfg.json'));

% provenance, a flag and not an assert: the solve is done and must not be lost
vs_ok = true;
for vs_n = {'userOpts', 'vehParams', 'vehModel', 'MLTP_initial', 'MLTP_solve', 'Powertrain'}
    vs_p = which(vs_n{1});
    fprintf('vsolve: which %-13s -> %s\n', vs_n{1}, vs_p);
    vs_ok = vs_ok && startsWith(lower(vs_p), lower(pwd));
end
vs_circ = eval('circuit');   % eval: a bare name could resolve to the RF Toolbox function 'circuit'
vs_ok = vs_ok && ischar(vs_circ) && strcmp(vs_circ, vs_cfg.circuit);

% IPOPT stats of the 23-state solve and of the 7-state init (vsolve_init.mat), reduced to the scalars asked for
vs_st = solver.stats();
vs_in = load('vsolve_init.mat');
vs_pat = '^(iter_count|return_status|success|t_wall_total|t_proc_total|t_wall_nlp_.*|n_call_nlp_.*)$';
vs_src = {vs_st, vs_in.stats};
vs_red = cell(1, 2);
for vs_i = 1:2
    vs_f = fieldnames(vs_src{vs_i});
    vs_red{vs_i} = rmfield(vs_src{vs_i}, vs_f(cellfun('isempty', regexp(vs_f, vs_pat, 'once'))));
end

try
    vs_pr = System.Diagnostics.Process.GetCurrentProcess();
    vs_peak = double(vs_pr.PeakWorkingSet64);
catch vs_e
    vs_peak = NaN;
    fprintf('vsolve: peak working set unavailable: %s\n', vs_e.message);
end

vs_it = vs_st.iterations;
vs_out = struct('w_opt', full(sol.x), 'lam_g', full(sol.lam_g), 'lam_x', full(sol.lam_x), 'f', full(sol.f), ...
                'x_opt', data.x_opt, 'u_opt', data.u_opt, 'xc_opt', data.xc_opt, 't_opt', data.t_opt, 'lap', data.t_opt(end), ...
                's_knot', s_knot, 'k_knot', k_knot, 'N', N, 'OPT_ds', OPT_ds, 'n_w', numel(w0), 'n_g', numel(lbg), ...
                'stats', vs_red{1}, 'iterations', vs_it, 'elapsedTime', elapsedTime, 'ipopt', opts.ipopt, ...
                'peak_ws_bytes', vs_peak, 'matlab_version', version, 'casadi_version', char(casadi.CasadiMeta.version()), ...
                'provenance_ok', vs_ok);
vs_out.elapsedTime_labels = {'start', 'transcription incl. vehModel.m (MLTP.m:387)', 'solver call (MLTP.m:394)'};
vs_out.init = struct('stats', vs_red{2}, 'elapsedTime', vs_in.elapsedTime(1:3), 'lap', vs_in.lap);
save(vs_cfg.mat_file, '-v7', '-struct', 'vs_out');

vs_j = rmfield(vs_out, {'w_opt', 'lam_g', 'lam_x', 'x_opt', 'u_opt', 'xc_opt', 't_opt', 's_knot', 'k_knot', 'iterations'});
vs_j.circuit = vs_cfg.circuit;
vs_j.inf_pr = vs_it.inf_pr(end);
vs_j.inf_du = vs_it.inf_du(end);
vs_fid = fopen(vs_cfg.json_file, 'w');
fwrite(vs_fid, jsonencode(vs_j, 'PrettyPrint', true, 'ConvertInfAndNaN', false));
fclose(vs_fid);

fprintf('vsolve: wrote %s (n_w %d, n_g %d, lap %.6f s, %d iterations, %s, peak working set %.2f GB)\n', ...
        vs_cfg.mat_file, vs_out.n_w, vs_out.n_g, vs_out.lap, vs_out.stats.iter_count, vs_out.stats.return_status, vs_peak / 2^30);
