%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cfile - File utils
%
%   Functions:
%       ct = load_text(fname, s_sep)    - load text file into cell array
%       save_text(fname, ct, s_sep)     - save text cell array to text file

%       save_darray(fname, da)          - save array to binary file
%       da = load_darray(fname)         - load array from binary file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef cfile
  
methods (Static)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ct = load_text(fname, s_sep)
    % Read a text file which contain file list
    %   The format of file list is: <col 1> <col 2> ... <col m>
    %
    % Parameters:
    %   fname - file list name
    %   s_sep - seperatio token (ex. ' ')
    %
    % Output:
    %   ct    - file list cell (nxn) (<col 1> <col 2> ... <col m>)
    %

    if( nargin < 2 ) 
        s_sep = ' ';
    end

    fid = fopen(fname, 'rt');

    n = 0;
    ct = {};

    while feof(fid) == 0
        s = strtrim(fgets(fid));

        % skip blank lines
        if length(s) < 1
            continue
        end

        % skip comment line
        if s(1) == '#'
            continue;
        end

        % split name/value
        a = cstring.str_split(s, s_sep);
        if length(a) >= 1
            n = n + 1;
            for j=1:length(a)
                ct{n, j} = strtrim(a{j});
            end
        end
    end

    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function save_text(fname, ct, s_sep)
    % Write a text file which contain nxm text field
    %   The format of file list is: <col 1> <col 2> ... <col m>
    %
    % Parameters:
    %   fname - file list name
    %   ct    - text cell array
    %   s_sep - seperatio token (ex. ' ')
    %
    % Output:
    %   ct    - file list cell (nxn) (<col 1> <col 2> ... <col m>)
    %

    if( nargin < 3 ) 
        s_sep = ' ';
    end

    fid = fopen(fname, 'wt');
    if( fid < 0 ) 
        fprintf('ERR: can not open file for write! %s\n', fname);
        return
    end
    
    [n, m] = size(ct);
    
    for i=1:n
        for j=1:m
            fprintf(fid, '%s', ct{i, j});
            fprintf(fid, '%s', s_sep);
        end
        
        fprintf(fid, '\n');
    end

    fclose(fid);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function save_struct(fname, s)
    % Write a text file of struct s
    %   The format of file list is: s.key = s.val
    %
    % Parameters:
    %   fname - file list name
    %   s     - struct 
    %
    % Output:
    %   ct    - file list is: s.key = s.val
    %

    fid = fopen(fname, 'wt');
    if( fid < 0 ) 
        fprintf('ERR: can not open file for write! %s\n', fname);
        return
    end

    c = struct2cell(s);
    f = fieldnames(s);

    for i=1:numel(c)
        fprintf(fid, '%s = %s\n', f{i}, c{i});
    end

    fclose(fid);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function save_darray(fname, da)
    % Write a binary data file
    %
    % Parameters:
    %   fname - binary data file name
    %   da    - data array
    %

    da_s = size(da);
    if( length(da_s) > 2 ) 
        error('ERR: unsupport 3-dimension (or above) array\n');
        return;
    end
    
    n = da_s(1);
    m = da_s(2);
    
    fid = fopen(fname, 'wb');
    if( fid < 0 ) 
        error('ERR: can not open file for write! %s\n', fname);
        return
    end
    
    fwrite(fid, n, 'uint64');
    fwrite(fid, m, 'uint64');
    d = reshape(da', 1, n*m);
    fwrite(fid, d, 'double');
    
    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function da = load_darray(fname)
    % Read a binary data file
    %
    % Parameters:
    %   fname - binary data file name
    %
    
    fid = fopen(fname, 'rb');
    if( fid < 0 ) 
        error('ERR: can not open file for reading! %s\n', fname);
        return
    end
    
    n = fread(fid, 1, 'uint64');
    m = fread(fid, 1, 'uint64');
    d = fread(fid, n*m, 'double');
    fclose(fid);
    
    da = reshape(d, m, n)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function save_farray(fname, da)
    % Write a float matrix to file
    %   (row first)
    %
    % Parameters:
    %   fname - binary data file name
    %   da    - data array
    %

    da_s = size(da);
    if( length(da_s) > 2 ) 
        error('ERR: unsupport 3-dimension (or above) array\n');
        return;
    end
    
    n = da_s(1);
    m = da_s(2);
    
    fid = fopen(fname, 'wb');
    if( fid < 0 ) 
        error('ERR: can not open file for write! %s\n', fname);
        return
    end
    
    fwrite(fid, n, 'uint64');
    fwrite(fid, m, 'uint64');
    d = reshape(da', 1, n*m);
    fwrite(fid, d, 'float');
    
    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function da = load_farray(fname)
    % Read a binary data file
    %
    % Parameters:
    %   fname - binary data file name
    %
    
    fid = fopen(fname, 'rb');
    if( fid < 0 ) 
        error('ERR: can not open file for reading! %s\n', fname);
        return
    end
    
    n = fread(fid, 1, 'uint64');
    m = fread(fid, 1, 'uint64');
    d = fread(fid, n*m, 'float');
    fclose(fid);
    
    da = reshape(d, m, n)';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function save_farray3d(fname, da)
    % Write a float matrix to file
    %   NOTE: according C array storage order
    %
    % Parameters:
    %   fname - binary data file name
    %   da    - data array (row x col x n)
    %

    da_s = size(da);
    if( length(da_s) ~= 3 ) 
        error('ERR: only support 3-dimension array\n');
        return;
    end
    
    row = da_s(1);
    col = da_s(2);
    n   = da_s(3);
    
    fid = fopen(fname, 'wb');
    if( fid < 0 ) 
        error('ERR: can not open file for write! %s\n', fname);
        return
    end
    
    fwrite(fid, col, 'uint64');
    fwrite(fid, row, 'uint64');
    fwrite(fid, n,   'uint64');
    d = reshape(permute(da, [2, 1, 3]), 1, row*col*n);
    fwrite(fid, d, 'float');
    
    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function da = load_farray3d(fname)
    % Read a binary data file
    %   NOTE: the file storge order is C array order
    %
    % Parameters:
    %   fname - binary data file name
    %
    
    fid = fopen(fname, 'rb');
    if( fid < 0 ) 
        error('ERR: can not open file for reading! %s\n', fname);
        return
    end
    
    col = fread(fid, 1, 'uint64');
    row = fread(fid, 1, 'uint64');
    n   = fread(fid, 1, 'uint64');
    d   = fread(fid, col*row*n, 'float');
    fclose(fid);
    
    da = permute(reshape(d, [col, row, n]), [2, 1, 3]);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function save_farrayEx(fname, da)
    % Write a float matrix to file
    %   NOTE: according Matlab array storage order
    %
    % Parameters:
    %   fname - binary data file name
    %   da    - data array
    %

    da_s = size(da);
    if( length(da_s) < 1 ) 
        error('ERR: only support 2-dimension or above array\n');
        return;
    end
      
    fid = fopen(fname, 'wb');
    if( fid < 0 ) 
        error('ERR: can not open file for write! %s\n', fname);
        return
    end
    
    fwrite(fid, length(da_s), 'uint64');            % write dimension number
    for i=1:length(da_s)
        fwrite(fid, da_s(i), 'uint64');             % write each dimension size
    end
    
    d = reshape(da, 1, prod(da_s));
    fwrite(fid, d, 'float');                        % write data
    
    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function da = load_farrayEx(fname)
    % Read a binary data file
    %   NOTE: the file storge order is Matlab array order
    %
    % Parameters:
    %   fname - binary data file name
    %
    
    fid = fopen(fname, 'rb');
    if( fid < 0 ) 
        error('ERR: can not open file for reading! %s\n', fname);
        return
    end
    
    dimn = fread(fid, 1, 'uint64');                 % read dimension number
    da_s = zeros(1, dimn);

    for i=1:dimn
        da_s(i) = fread(fid, 1, 'uint64');          % read each dimension size
    end

    d   = fread(fid, prod(da_s), 'float');
    fclose(fid);
    
    da = reshape(d, da_s);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function saveArray(fname, da, datatype)
    % Write a float matrix to file
    %   NOTE: according Matlab array storage order
    %
    % Parameters:
    %   fname       - binary data file name
    %   da          - data array
    %   datatype    - data type ('double', 'float' (default), 
    %                   'uint64', 'uint32', 'uint16', 'uint8',
    %                   'int64', 'int32', 'int16', 'int8')
    %

    if nargin <= 2 
        datatype = 'float';
    end

    da_s = size(da);
    if( length(da_s) < 1 ) 
        error('ERR: only support 2-dimension or above array\n');
        return;
    end

    % convert data type
    if( strcmp(datatype, 'double') == 1 )
        da = double(da);
    else if( strcmp(datatype, 'float') == 1 ) 
        da = single(da);
    else if( strcmp(datatype, 'uint64') == 1 ) 
        da = uint64(da);
    else if( strcmp(datatype, 'uint32') == 1 ) 
        da = uint32(da);
    else if( strcmp(datatype, 'uint16') == 1 ) 
        da = uint16(da);
    else if( strcmp(datatype, 'uint8') == 1 ) 
        da = uint8(da);
    else if( strcmp(datatype, 'int64') == 1 ) 
        da = int64(da);
    else if( strcmp(datatype, 'int32') == 1 ) 
        da = int32(da);
    else if( strcmp(datatype, 'int16') == 1 ) 
        da = int16(da);
    else if( strcmp(datatype, 'int8') == 1 ) 
        da = int8(da);
    end

    % open file  
    fid = fopen(fname, 'wb');
    if( fid < 0 ) 
        error('ERR: can not open file for write! %s\n', fname);
        return
    end
    
    % write dimN and dims
    fwrite(fid, length(da_s), 'uint64');            % write dimension number
    for i=1:length(da_s)
        fwrite(fid, da_s(i), 'uint64');             % write each dimension size
    end
    
    % write data
    d = reshape(da, 1, prod(da_s));
    fwrite(fid, d, datatype);                       % write data
    
    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function da = loadArray(fname, datatype)
    % Read a binary data file
    %   NOTE: the file storge order is Matlab array order
    %
    % Parameters:
    %   fname - binary data file name
    %   datatype    - data type ('double', 'float' (default), 
    %                   'uint64', 'uint32', 'uint16', 'uint8',
    %                   'int64', 'int32', 'int16', 'int8')
    %

    if nargin <= 1 
        datatype = 'float';
    end
    
    fid = fopen(fname, 'rb');
    if( fid < 0 ) 
        error('ERR: can not open file for reading! %s\n', fname);
        return
    end
    
    dimn = fread(fid, 1, 'uint64');                 % read dimension number
    da_s = zeros(1, dimn);

    for i=1:dimn
        da_s(i) = fread(fid, 1, 'uint64');          % read each dimension size
    end

    d   = fread(fid, prod(da_s), datatype);
    fclose(fid);
    
    da = reshape(d, da_s);
end


end % end of methods
end % end of classdef

