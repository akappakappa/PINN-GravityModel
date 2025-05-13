mtrx  = readmatrix("src/data/Model/eros_shape_200700.obj", 'FileType', 'text', 'OutputType', 'string');
faces = double(mtrx(contains(mtrx(:, 1), 'f'), 2:4));
verts = double(mtrx(contains(mtrx(:, 1), 'v'), 2:4));
verts = verts / max(max(abs(verts)));
shape = triangulation(faces, verts);
points = [0,0,0; 1,1,1; 0.1,0.1,0.1; 2,2,2];
inside = inpolyhedron(shape, points);

disp("Breakpoint");

function inside = inpolyhedron(shape, points)
    nFaces   = size(shape.ConnectivityList, 1);
    nPoints  = size(points, 1);

    % Ray-casting algorithm to check if points are inside a 3D shape
    epsilon  = 1e-10;
    inside   = false(nPoints, 1);
    rayDir   = [1, 0, 0];
    hit      = NaN(nFaces, nPoints); % Preallocation
    
    % Parallel check
    normals          = faceNormal(shape);
    parallel         = abs(dot(repmat(rayDir, nFaces, 1), normals, 2)) < epsilon;
    hit(parallel, :) = fillmissing(hit(parallel, :), 'constant', false);
    
    % Barycentric coordinates of each point in points w.r.t. each face
    barycentrics = NaN(nFaces, 3, nPoints); % Preallocation
    for i = 1:nFaces
        barycentrics(i, :, :) = cartesianToBarycentric(shape, repmat(i, nPoints, 1), points).';
    end
    u = barycentrics(:, 1, :);
    v = barycentrics(:, 2, :);
    mask = squeeze(u < 0 | u > 1 | v < 0 | u + v > 1);
    hit(mask) = fillmissing(hit(mask), 'constant', false);

    % Intersection distance t
    for i = 1:nFaces
        face = shape.ConnectivityList(i, :);
        edge1 = shape.Points(face(2), :) - shape.Points(face(1), :);
        edge2 = shape.Points(face(3), :) - shape.Points(face(1), :);
        h     = cross(rayDir, edge2);
        a     = dot(edge1, h);
        if a > -epsilon && a < epsilon
            % If a is near zero, ray is parallel to triangle => no hit
            continue;
        end
        f = 1 / a;
        s = points - shape.Points(face(1), :);
        u = f * dot(s, h);
        if u < 0 || u > 1
            continue;
        end
        q = cross(s, edge1);
        v = f * dot(rayDir, q);
        if v < 0 || u + v > 1
            continue;
        end
        t   = f * dot(edge2, q);
        hit(i, :) = t > epsilon; % If t is positive, ray intersects triangle
    end
    



end


function inside = inShapeModel(shape, points)
    % Check if points are inside the shape model
    faces    = shape.ConnectivityList;
    vertices = shape.Points;
    inside   = false(size(points, 1), 1);   % Preallocate boolean flags
    rayDir   = [1, 0, 0];                   % Ray direction (any works)

    % Ray-casting algorithm
    for i = 1:size(points, 1)
        point              = points(i, :);
        intersectionsCount = 0;

        % Check each face
        for j = 1:size(faces, 1)
            face = vertices(faces(j, :), :);
            hit  = algMollerTrumbore(point, rayDir, face(1, :), face(2, :), face(3, :));
            if hit
                intersectionsCount = intersectionsCount + 1;
            end
        end

        % If odd number of intersections, point is inside the shape
        inside(i) = mod(intersectionsCount, 2) == 1;
    end


    function hit = algMollerTrumbore(point, rayDir, v1, v2, v3)
        epsilon = 1e-10;

        % Parralel check
        edge1 = v2 - v1;
        edge2 = v3 - v1;
        h     = cross(rayDir, edge2);
        a     = dot(edge1, h);
        if a > -epsilon && a < epsilon
            % If a is near zero, ray is parallel to triangle => no hit
            hit = false;
            return;
        end

        % Barycentric coordinates (u, v)
        f = 1 / a;

        % Calculate u
        s = point - v1;
        u = f * dot(s, h);
        if u < 0 || u > 1
            % If u is outside [0, 1], point is outside triangle
            hit = false;
            return;
        end

        % Calculate v
        q = cross(s, edge1);
        v = f * dot(rayDir, q);
        if v < 0 || u + v > 1
            % If v is outside [0, 1] or u + v > 1, point is outside triangle
            hit = false;
            return;
        end

        % Calculate t
        t   = f * dot(edge2, q);
        hit = t > epsilon; % If t is positive, ray intersects triangle
    end
end