mtrx  = readmatrix("src/data/Model/eros_shape_200700.obj", 'FileType', 'text', 'OutputType', 'string');
faces = double(mtrx(contains(mtrx(:, 1), 'f'), 2:4));
verts = double(mtrx(contains(mtrx(:, 1), 'v'), 2:4));
verts = verts / max(max(abs(verts)));
shape = triangulation(faces, verts);

disp("Breakpoint");

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