disp("a");

function shape = readShapeModel(fname)
    % Setup vertices and faces
    shape              = struct();
    [shape.v, shape.f] = deal([]);

    % Parse the OBJ file
    fid = fopen(fname, 'r');
    while ~feof(fid)
        line = fgetl(fid);
        switch line(1)
            case 'v'    % Vertex: v x y z
                shape.v = [shape.v; sscanf(line(2:end), '%f')'];
            case 'f'    % Face: f v1 v2 v3
                shape.f = [shape.f; sscanf(line(2:end), '%d')'];
            otherwise   % Ignore
        end
    end
    fclose(fid);

    % Check if the shape is valid
    if isempty(shape.v) || isempty(shape.f)
        error("[ERR] Invalid Shape Model");
    end
end


function shape = normalizeShapeModel(shape)
    shape.v = shape.v / max(max(abs(shape.v)));
end