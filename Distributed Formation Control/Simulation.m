%%%%%%%%%%%%%Main Simulation of Simplified 2D version of 4.2.2
%%initialization
obstacle1 = [1, 2, 2, 1;
             0, 0, .45, .45]; %obstacle 1 vertices
obstacle2 = [1, 2, 2, 1;
            .55, .55, 1, 1]; %obstacle 2 vertices
Wf_configuration2x2 = [-0.1, -0.1, 0.1, 0.1;
                        -0.1, 0.1, -0.1, 0.1]; %vector from center
C_2x2 = 1; %configuration weight
Wf_configuration1x4 = [-.15, -.05, .05, .15;
                          0,    0,   0,   0]; %vector from center
C_1x4 = 2; %configuration weight
c_init = [0.5; 0.5]; %initial center of mass
plot_position = c_init + Wf_configuration2x2; %plotting function holdover
wt = 1; %translational weight
ws = 10; %scaling weight
wq = 1; %rotational weight (not used here)
stage = 1; %stage iteration counter for discritization of simulation
goal_config = [1.5, 0.5; 1.5, 0.5; 2.5, 0.5; 2.5, 0.5];

    %%%%%%% Hand Calculation approximations of PCi per 3.3
    PCi(1).vertices = [0.4, 0; 0.4, 1; 1, 1; 1, 0];
    PCi(2).vertices = [0.65, .45; 0.65, 0.55; 2, 0.55; 2, 0.45];
    PCi(3).vertices = [1.3, 0.45; 1.3, 0.55; 2.3, .55; 2.3, 0.45];
    PCi(4).vertices = [2, 0; 2, 1; 2.65, 1; 2.65, 0];
    
    %%%%%% Hand Calculation approximations of Pci per 3.3
    Pci(1).vertices = [.4, .45; .4, .55; 2, .55; 2, .45];
    Pci(2).vertices = [.4, .45;.4, .55; 3, .55; 3, .45 ];
    Pci(3).vertices = [.4, .45; .4, .55; 3, .55; 3, .45];
    Pci(4).vertices = [2, 0; 2, 1; 3, 1; 3, 0 ];
    
    %%%%%%% Compute P_i
    %IntersectionHull Library (Library functions in UtilityFunctions folder)
    %Matt J (2024). Analyze N-dimensional Convex Polyhedra
    %(https://www.mathworks.com/matlabcentral/fileexchange/30892-analyze-n-dimensional-convex-polyhedra),
    %MATLAB Central File Exchange. Retrieved March 27, 2024.
    
    for i = 1:4
        PC(i) = intersectionHull('vert', PCi(i).vertices, 'vert', Pci(i).vertices);
    end


%%%%%%%%%%%% Begin Simulation    
%%%%%%%%%%%% Compute Optimal Formation
figure(3)
while stage < 5
    fun1 = @(X)wt*norm([X(1); X(2)] - (goal_config(stage, :)'-c_init), 2)^2+ws*norm(X(3) - 1, 2)^2+wq*norm(1-1, 2)^2 + C_2x2; %define cost function
    fun2 = @(X)wt*norm([X(1); X(2)] - (goal_config(stage, :)'-c_init), 2)^2+ws*norm(X(3) - 1, 2)^2+wq*norm(1-1, 2)^2 + C_1x4; 
    fcon_2x2 = @(X)NonLinConst1(X, c_init, Wf_configuration2x2,  PC(stage)); %Passby function (for passing in extra variables to nonlinear constraints)
    fcon_1x4 = @(X)NonLinConst2(X, c_init, Wf_configuration1x4,  PC(stage));
    options = optimoptions("fmincon",...
        "Algorithm","interior-point",...
        "EnableFeasibilityMode",true,...
        "SubproblemAlgorithm","cg", 'Display', 'off');
    [X1, FVAL1, exitflag1, output1] = fmincon(fun1, [0, 0, 1], [], [], [], [], [-inf, -inf, 1], [], fcon_2x2, options); %optimization function call
    [X2, FVAL2, exitflag2, output2] = fmincon(fun2, [0, 0, 1], [], [], [], [], [-inf, -inf, 1], [], fcon_1x4, options);
    
    if exitflag1 < 0
        c_init = c_init + [X2(1); X2(2)]; %formation 1 is unfeasible
        plot_position = [plot_position c_init+Wf_configuration1x4];
    elseif FVAL1<FVAL2
        c_init = c_init + [X1(1); X1(2)]; %formation 1 has less energy
        plot_position = [plot_position c_init+Wf_configuration2x2];
    else
        c_init = c_init + [X2(1); X2(2)]; %formation 2 has less energy
        plot_position = [plot_position c_init+Wf_configuration1x4];
    end
    
    %Plot functions
    subplot(4, 1, stage)
    hold on
    xlim([0 3])
    ylim([0, 1])
    box on
    fill(obstacle1(1, :), obstacle1(2, :), [0, 0, 0])
    fill(obstacle2(1, :), obstacle2(2, :), [0, 0, 0])

    fill(PCi(stage).vertices(:,1)', PCi(stage).vertices(:,2)', [0, 0, 1], 'FaceAlpha', .5)
    fill(Pci(stage).vertices(:,1)', Pci(stage).vertices(:,2)', [1, 1, 0], 'FaceAlpha', .3)
    
    plot(plot_position(1, (stage-1)*4+1), plot_position(2, (stage-1)*4+1), 'ko')
    plot(plot_position(1, (stage)*4+1), plot_position(2, (stage)*4+1), 'ro')
    plot(linspace(plot_position(1, (stage-1)*4+1), plot_position(1, stage*4+1), 10), linspace(plot_position(2, (stage-1)*4+1), plot_position(2, stage*4+1),10), 'gx')
    
    plot(plot_position(1, (stage-1)*4+2), plot_position(2, (stage-1)*4+2), 'ko')
    plot(plot_position(1, (stage)*4+2), plot_position(2, (stage)*4+2), 'ro')
    plot(linspace(plot_position(1, (stage-1)*4+2), plot_position(1, stage*4+2), 10), linspace(plot_position(2, (stage-1)*4+2), plot_position(2, stage*4+2),10), 'gx')
    
    plot(plot_position(1, (stage-1)*4+3), plot_position(2, (stage-1)*4+3), 'ko')
    plot(plot_position(1, (stage)*4+3), plot_position(2, (stage)*4+3), 'ro')
    plot(linspace(plot_position(1, (stage-1)*4+3), plot_position(1, stage*4+3), 30), linspace(plot_position(2, (stage-1)*4+3), plot_position(2, stage*4+3),30), 'gx')

    plot(plot_position(1, (stage-1)*4+4), plot_position(2, (stage-1)*4+4), 'ko')
    plot(plot_position(1, (stage)*4+4), plot_position(2, (stage)*4+4), 'ro')
    plot(linspace(plot_position(1, (stage-1)*4+4), plot_position(1, stage*4+4), 30), linspace(plot_position(2, (stage-1)*4+4), plot_position(2, stage*4+4),30), 'gx')
    title(append('Trajectory as calculated during Stage ', string(stage)))
    xlabel('x position [m]')
    ylabel('y position [m]')

    stage = stage+1; %advance to next stage
end

%%%% Robot assignment is trivial in this example and is therefore skipped.
%%%% See Burger et al. (2012) for algorithm referenced in paper.

%%%%%%%%%% Nonlinear constraint functions for convex hull PC
function [c,ceq] = NonLinConst1(X, c_init, Wf_configuration,  PC)  %nonlinear constraint function for optimization
c = PC.lcon{1}*([X(1); X(2)] + (c_init+X(3).*Wf_configuration))-PC.lcon{2}; %robots contained in convex hull PC
ceq = []; %no equality constraints
end
function [c,ceq] = NonLinConst2(X, c_init, Wf_configuration,  PC)  %nonlinear constraint function for optimization
c = PC.lcon{1}*([X(1); X(2)] + (c_init+X(3).*Wf_configuration))-PC.lcon{2}; %robots contained in convex hull PC
ceq = []; %no equality constraints
end
