tmpDir = pwd;
cd(fileparts(which('compile_graph_loss_d1_simplex_mex.m')));
CXXFLAGScat = '-fopenmp';
LDFLAGScat = '-fopenmp';
try
    depList = {'PFDR_graph_loss_d1_simplex', 'proj_simplex_metric'};
    compile_Cpp_mex('PFDR_graph_loss_d1_simplex', depList, CXXFLAGScat, LDFLAGScat, true);
catch
	cd(tmpDir);
	rethrow(lasterror);
end
cd(tmpDir);
