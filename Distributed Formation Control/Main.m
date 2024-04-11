% Main.m - Mock implementation of example from Distributed Multi-Robot
% Formation Control In Dynamic Environments by J. Alonso-Mora, et. all.
% (2018). DOI:s10514-018-9783-9

%Example code by Michael Klinefelter, BYU (2024)

%Minimum Compilation instructions:
%The optimization toolboxe is required to run
%this code.  Installation can be done under 
%Home -> Environment -> Add-ons ->Manage Add-ons.

%Main.m is a container of the individual steps in the algorithm displayed
%in Figure 4 and necessary set-up variables to run a scaled-down version of
%4.1.1 - the convex hull problem and 4.2.2 - the multi-drone formation
%control. 

%Simulation.m runs a simplified version of example 4.2.2 from the 
%aforementioned paper. It is run from Main.m or can be run independantly
%from the file.

%Utility functions: Matt J (2024). Analyze N-dimensional Convex Polyhedra
% (https://www.mathworks.com/matlabcentral/fileexchange/30892-analyze-n-dimensional-convex-polyhedra)
% MATLAB Central File Exchange. Retrieved March 24, 2024. See license.txt
% for more information.

clear
clc
rng(0, "twister")

%Section 1 - Convex Hull of Robot Positions (consensus)
    %Part 4.1.1 - Random Robot positions num_robots = 100 in 10x10 cube with
    %communication radius of 3.5.
    num_robots = 100;
    Robot_For_ConvHull = struct('position', [], 'connectivity', [], 'convexhull', [], 'newconvexhullpoints', [],...
        'receive_communication', [], 'theta_eval', [], 'max_theta_eval', []);
    for i = 1:num_robots
        Robot_For_ConvHull(i).position = rand(1, 3).*10;  %Random initial Positions
        Robot_For_ConvHull(i).connectivity = [];           %Inital Connectivity is empty
        Robot_For_ConvHull(i).convexhull = Robot_For_ConvHull(i).position;  %initialize convex hull to itself.
        Robot_For_ConvHull(i).newconvexhullpoints = Robot_For_ConvHull(i).position; %initialize new convex hull points to send to others.
    end
    for i = 1:num_robots-1
        for j = i+1:num_robots
            if norm(Robot_For_ConvHull(i).position - Robot_For_ConvHull(j).position , 2) < 3.5 %Establish connectivity with Radius 1
                Robot_For_ConvHull(i).connectivity(end+1) = j;
                Robot_For_ConvHull(j).connectivity(end+1) = i;                
            end
        end
    end
    G = zeros(num_robots, num_robots); %Initialize adjacency matrix
    for i = 1:num_robots
        for j = Robot_For_ConvHull(i).connectivity
            G(i, j) = 1; % Create adjacency Matrix
        end
    end
    G = graph(G);  %matlab requires graph Struct for efficient computation.
    d = max(distances(G), [], 'all'); %compute diameter of undirected graph.
    tic
    for i = 1:d-1 %D steps required for convergence of convex hulls by information sharing.
        for j = 1:num_robots
            for k = Robot_For_ConvHull(j).connectivity
                Robot_For_ConvHull(k).receive_communication(end+1:end+size(Robot_For_ConvHull(j).newconvexhullpoints, 1),:) = Robot_For_ConvHull(j).newconvexhullpoints;
            end
        end
        for j=1:num_robots  %distribute (up to limit of running hardware)
            Robot_For_ConvHull(j).newconvexhullpoints = [];
            [Robot_For_ConvHull(j).convexhull, Robot_For_ConvHull(j).newconvexhullpoints] = ConvexHull(Robot_For_ConvHull(j).convexhull, Robot_For_ConvHull(j).receive_communication);
            Robot_For_ConvHull(j).receive_communication = [];
        end
    end
    t = toc;
    disp(append('Total time for ', int2str(num_robots), ' robots to serially converge to convex hull after ', int2str(d), ' iterations was ', string(t), ' seconds'))
    graphindices = randperm(num_robots, 4);
    figure(1)
    title('hello')
    subplot(2, 2, 1)
    for i = graphindices
        nexttile
        Indices = convhull(Robot_For_ConvHull(i).convexhull, 'Simplify', true);
        trisurf(Indices, Robot_For_ConvHull(i).convexhull(:,1), Robot_For_ConvHull(i).convexhull(:,2), Robot_For_ConvHull(i).convexhull(:,3), 'FaceColor','cyan')
        axis equal
        title(append('Convex Hull as determined by Robot #', int2str(i)))
        xlabel('x')
        ylabel('y')
        zlabel('z')
    end

%Section 3.2 - Prefered Direction of Motion
   theta_dist = rand(1, 8).*2; %generate random 'heading's
    for i = 1:num_robots %Initialize evaluation of prefered heading 
                        %(random in this example, would need
                        %to replace with a utility function)
        Robot_For_ConvHull(i).theta_eval = rand(1, 8).*theta_dist;
    end
    for i = 1:d-1 %D steps required for consensus of Prefered Direction of Motion
        for j = 1:num_robots
            for k = Robot_For_ConvHull(j).connectivity
                Robot_For_ConvHull(k).receive_communication(end+1, :) = Robot_For_ConvHull(j).theta_eval;
            end
        end
        for j=1:num_robots  
            [Robot_For_ConvHull(j).theta_eval] = PreferredMotion(Robot_For_ConvHull(j).theta_eval, Robot_For_ConvHull(j).receive_communication);
            Robot_For_ConvHull(j).receive_communication = [];
            Robot_For_ConvHull(j).max_theta_eval = max(Robot_For_ConvHull(j).theta_eval);  %take the max (technically this only needs to happen at the end, 
                                                                                           % but i put it here to graph the response over time.
                                                                                           %Additionally, the index is more important here (max() is just the utility)
            maxConvergence(j, i) = Robot_For_ConvHull(j).max_theta_eval;
        end
    end
    figure(2)
    plot(maxConvergence')
    xlabel('Iteration')
    ylabel('arg max u_i')
    title('Maximum Utility of Preferred Direction of Motion of each Robot by Iteration')
    xticks([1, 2, 3, 4, 5])

    
%Remaining sections shown as output of Section 4.2.2
%See Simulation.m for details
run("Simulation.m")

