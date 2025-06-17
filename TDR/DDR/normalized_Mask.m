function M=normalized_Mask(var,i,j)
    m=-var+(var-(-var)).*rand(i,j);
    M=m./norm(m);
end