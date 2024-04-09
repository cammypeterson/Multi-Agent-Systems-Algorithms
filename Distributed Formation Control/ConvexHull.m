function [newConvexHull, newPointsNotInOriginal] = ConvexHull(convexhull, newpoints)
    %Computes convex hull from collection of points.
    %newCovexHull = combined convex hull of vertices in convexhull and
    %newpoints.
    %newPointsNotInOriginalHull = vertices in newConvexHull not in
    %convexhull.  Used for transmission between robots in next iteration to
    %save computational time.
    newpoints= uniquetol(newpoints,'ByRows',true); %Remove duplicates from neighboring nodes
    
    %initialize arrays
    combinedPoints = [convexhull; newpoints];
    newConvexHull = []; 
    newPointsNotInOriginal = [];

    try
        Indices = convhull(combinedPoints, 'Simplify', true); %compute convex hull
        i = unique(Indices(:));
        newConvexHull = combinedPoints(i, :); %Reduce convex hull to unique indices to save memory
        for j = 1:size(newpoints, 1)
            if ismembertol(newpoints(j, :), newConvexHull,'ByRows',true)
                newPointsNotInOriginal(end+1, :) = newpoints(j, :); %Reduce communication protocol to only include new points
            end
        end
    catch %This Catch is due to MATLABs implementation of convhull.m which 
          % defines points or planes in R3 as not having convex hulls.
          %This block of code propogates the points forward if not enough
          %unique points for a 3d structure is defined which aligns with
          %the standard definition that a point/plane is its own convex
          %hull.
        newConvexHull = combinedPoints;
        for j = 1:size(newpoints, 1)
            if ismembertol(newpoints(j, :), newConvexHull,'ByRows',true)
                newPointsNotInOriginal(end+1, :) = newpoints(j, :);
            end
        end
    end
    newPointsNotInOriginal = uniquetol(newPointsNotInOriginal,'ByRows',true); %check for/remove duplicates
end
