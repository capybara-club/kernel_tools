from .csrc.wrapper import mgSyevd, syevdx

def cusolver_eigh(
        a,
        overwrite_a = False,
        subset_by_index = None, 
        lower = True, 
        eigvals_only = False
):
    """
    Uses cusolver cusolverDnXsyevdx to either get all the eigenvalues / eigenvectors
    of a system or a range of indices specified by subset_by_index. This function
    attempts to match the api of the scipy linalg.eigh function. The base call
    to cusolverDnXsyevdx overwrites the 'a' matrix and additionally allocates a workspace.
    For the eigenvalues a vector of size N is required even if the range requested is small.
    This is allocated by pytorch and passed in. The view of it in the range specified is cloned 
    and returned to save on memory. Similarly the 'a' matrix when overwrite_a is True and a range of 
    eigenvalues/eigenvectors is requested has its view of the range cloned. This is to
    allow PyTorch to deallocate the full square 'a' matrix after use. A view of this matrix
    would have PyTorch retain the full memory space.

    Parameters:
    a (torch::Tensor): The matrix
    overwrite_a: Avoids a copy of the a tensor when true
    subset_by_index: If provided, this two-element iterable defines the start and 
        the end indices of the desired eigenvalues (ascending order and 0-indexed).
    lower: Use the lower triangle, if false, use the upper triangle of the symmetric matrix
    eigvals_only: Only return the eigenvalues of a. If overwrite_a is True, then
        'a' will not be copied and this routine will still destroy 'a'.
    
    Returns:
    eigenvalues, eigenvectors of the range specified if eigvals_only is false
    eigenvalues of the range specified if eigvals_only is true
    """

    return syevdx(a, 
                  overwrite_a=overwrite_a, 
                  subset_by_index=subset_by_index, 
                  lower=lower, 
                  eigvals_only=eigvals_only
            )

def cusolver_mg_eigh(
    a,
    overwrite_a = False
):
    """
    Uses cusolver cusolverMgsyevd to get all of the eigenvalues and eigenvectors of
    a system. It uses all nvidia devices visible to split the compute work and share 
    the working space. The base call of this function overwrites the eigenvectors in 
    to the 'a' matrix. When 'overwrite_a' is True, the vectors are returned in to 
    the passed in 'a' matrix

    Parameters:
    a (torch::Tensor): The matrix
    overwrite_a: Avoids a copy of the a tensor when true
    
    Returns:
    eigenvalues, eigenvectors
    """
    return mgSyevd(a, overwrite_a=overwrite_a)
