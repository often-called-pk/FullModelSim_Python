% vexport.m - parameter + NLP export of the MATLAB original (validation gates 1 and 2, no solve).
%
% Driver of validation/matlab_batch.py, run in the BASE workspace right after MLTP_build.m
% (MLTP.m up to and including the nlpsol line, IPOPT max_iter 0):   matlab -batch "MLTP_build; vexport"
% MLTP.m clears the workspace, so the paths come from vexport_cfg.json (written by matlab_batch.py):
%   circuit, params_file, nlp_file, extra_w ('' or a .mat), seed.
% All own variables are prefixed vx_ so nothing of the model workspace is touched.
%
% params_file (JSON, same keys as validation/py_export.py): what the executed pipeline held.
%   vp        vehParams.m result before vehModel.m derives the aero block (snapshot vx_vp_pre)
%   aero_used Cl_fl Cl_fr Cl_rl Cl_rr Cd as vehModel.m evaluates them at the static wing angles
%   ipopt     userOpts.m options as shipped (opts_shipped, taken before the max_iter override)
% nlp_file (MAT v7): lbw ubw lbg ubg w0 w_rand, W = [w0 w_rand <columns of extra_w>] (n_w x k),
%   W_names (1 x k), F = f(W) (1 x k), G = g(W) (n_g x k), s_knot k_knot dsk.
%   w_rand = w0 + 0.05*range.*(2*rand-1) after rng(seed), infinite range -> 1, clipped to [lbw, ubw].
%   extra_w: every numeric variable with n_w rows adds its columns (labels: variable name, _j).
import casadi.*

vx_cfg = jsondecode(fileread('vexport_cfg.json'));

% provenance: every script must come from this copy, not from the MATLAB path
for vx_n = {'userOpts', 'vehParams', 'vehModel', 'MLTP_initial', 'Powertrain'}
    vx_p = which(vx_n{1});
    fprintf('vexport: which %-13s -> %s\n', vx_n{1}, vx_p);
    assert(startsWith(lower(vx_p), lower(pwd)), 'vexport: %s is not the copy in %s', vx_n{1}, pwd);
end

%% (a) parameters
vx = struct();
vx_circ = eval('circuit');   % eval: a bare name could resolve to the RF Toolbox function 'circuit'
assert(ischar(vx_circ) && strcmp(vx_circ, vx_cfg.circuit), 'vexport: workspace circuit differs from the patched one');
vx.circuit = vx_circ;
vx.AeroConfig = AeroConfig;  vx.ATD = ATD;  vx.Electric_4Motors = Electric_4Motors;
vx.TyreModel = TyreModel;  vx.tire = tire;  vx.vi = vi;  vx.ni = ni;
vx.vp = vx_vp_pre;
vx.pt = pt;
vx.mf = struct();
vx_who = who;
for vx_i = 1:numel(vx_who)
    if ~isempty(regexp(vx_who{vx_i}, '^[pr][A-Z][xy][0-9]$', 'once'))
        vx.mf.(vx_who{vx_i}) = eval(vx_who{vx_i});
    end
end
vx.CG_lin = struct('CG_h_deg_per_mm_linear', CG_h_deg_per_mm_linear, ...
                   'CG_r_deg_per_deg_linear', CG_r_deg_per_deg_linear, ...
                   'CG_p_deg_per_deg_linear', CG_p_deg_per_deg_linear);
vx.aero = aero;
vx.aero_used = struct('Cl_fl', vp.Cl_fl, 'Cl_fr', vp.Cl_fr, 'Cl_rl', vp.Cl_rl, 'Cl_rr', vp.Cl_rr, 'Cd', vp.Cd);
vx.Xi = Xi;  vx.Xf = Xf;
vx.OPT_ds = OPT_ds;  vx.OPT_d = OPT_d;  vx.OPT_e = OPT_e;  vx.OPT_uinter = OPT_uinter;
vx.N = N;  vx.nx = nx;  vx.nu = nu;  vx.n_w = numel(w0);  vx.n_g = numel(lbg);
vx.x_s = x_s;  vx.u_s = u_s;  vx.x_min = x_min;  vx.x_max = x_max;  vx.u_min = u_min;  vx.u_max = u_max;
vx.duk_lb = duk_lb;  vx.duk_ub = duk_ub;       % scaled (vehModel.m divides them by u_s)
vx.h_lb = h_lb;  vx.h_ub = h_ub;  vx.hnames = hnames;
vx.ru = ru;  vx.rdu = rdu;  vx.rdu2 = rdu2;
if exist('opts_shipped', 'var'), vx.ipopt = opts_shipped.ipopt; else, vx.ipopt = opts.ipopt; end
vx.track = struct('n', numel(track.s), 's_end', track.s(end), 'k_abssum', sum(abs(track.k)), 'k_sumsq', sum(track.k .^ 2));

vx_fid = fopen(vx_cfg.params_file, 'w');
fwrite(vx_fid, jsonencode(vx, 'PrettyPrint', true, 'ConvertInfAndNaN', false));
fclose(vx_fid);
fprintf('vexport: wrote %s\n', vx_cfg.params_file);

%% (b) NLP bounds, evaluation points, f and g
vx_w0 = w0(:);
rng(vx_cfg.seed);
vx_rg = ubw(:) - lbw(:);
vx_rg(~isfinite(vx_rg)) = 1;
vx_wr = min(max(vx_w0 + 0.05 * vx_rg .* (2 * rand(numel(vx_w0), 1) - 1), lbw(:)), ubw(:));
vx_W = [vx_w0, vx_wr];
vx_wn = {'w0', 'w_rand'};
if ~isempty(vx_cfg.extra_w)
    vx_ex = load(vx_cfg.extra_w);
    vx_fn = fieldnames(vx_ex);
    for vx_i = 1:numel(vx_fn)
        vx_v = vx_ex.(vx_fn{vx_i});
        assert(isnumeric(vx_v) && size(vx_v, 1) == numel(vx_w0), 'vexport: extra w "%s" needs %d rows', vx_fn{vx_i}, numel(vx_w0));
        for vx_j = 1:size(vx_v, 2)
            vx_W(:, end + 1) = vx_v(:, vx_j); %#ok<SAGROW>
            if size(vx_v, 2) == 1, vx_wn{end + 1} = vx_fn{vx_i}; else, vx_wn{end + 1} = sprintf('%s_%d', vx_fn{vx_i}, vx_j); end %#ok<SAGROW>
        end
    end
end

vx_fg = Function('vx_fg', {nlp.x}, {nlp.f, nlp.g}, {'w'}, {'f', 'g'});
vx_F = zeros(1, size(vx_W, 2));
vx_G = zeros(numel(lbg), size(vx_W, 2));
for vx_j = 1:size(vx_W, 2)
    [vx_f, vx_g] = vx_fg(vx_W(:, vx_j));
    vx_F(vx_j) = full(vx_f);
    vx_G(:, vx_j) = full(vx_g);
end

vx_out = struct('lbw', lbw, 'ubw', ubw, 'lbg', lbg, 'ubg', ubg, 'w0', vx_W(:, 1), 'w_rand', vx_W(:, 2), ...
                'W', vx_W, 'F', vx_F, 'G', vx_G, 's_knot', s_knot, 'k_knot', k_knot, 'dsk', dsk);
vx_out.W_names = vx_wn;
save(vx_cfg.nlp_file, '-v7', '-struct', 'vx_out');
fprintf('vexport: wrote %s (n_w %d, n_g %d, %d evaluation points: %s)\n', vx_cfg.nlp_file, numel(w0), numel(lbg), size(vx_W, 2), strjoin(vx_wn, ', '));
