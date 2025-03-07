from kernel_tools.csrc.wrapper import mgSyevd, getrf_workspace_query, getrf, getrs, syevdx, syevdx_workspace_query, syev_batched, syev_batched_workspace_query, mgSyevd_workspace_query

def cusolver_getrf_workspace_requirements(M, N, dtype):
    """
    Returns the workspace required bytes for cusolver cusolverDnXsyevdx for the 
    specified problem size and data type. Only torch.float32 and torch.float64 are
    supported.

    Parameters:
    N (torch::Tensor): The size of the symmetric matrix to extract eigenvalues/eigenvectors from
    dtype: The data type of the matrix
    
    Returns:
    workspaceInBytesDevice, workspaceInBytesHost
    """

    return getrf_workspace_query(M, N, dtype)

def cusolver_getrf(
        a,
        overwrite_a = False,
        verbose = False
):
    return getrf(a, 
                  overwrite_a=overwrite_a, 
                  verbose=verbose
            )

def cusolver_getrs(
        a,
        ipiv,
        targets,
        overwrite_targets = False,
        verbose = False
):
    return getrs(a, ipiv, targets,
                  overwrite_targets=overwrite_targets, 
                  verbose=verbose
            )

def cusolver_eigh(
        a,
        overwrite_a = False,
        subset_by_index = None, 
        lower = True, 
        eigvals_only = False,
        verbose = False
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
    verbose: Currently only prints the requested workspace size. This is useful if you are crashing.

    Returns:
    eigenvalues, eigenvectors of the range specified if eigvals_only is false
    eigenvalues of the range specified if eigvals_only is true
    """

    return syevdx(a, 
                  overwrite_a=overwrite_a, 
                  subset_by_index=subset_by_index, 
                  lower=lower, 
                  eigvals_only=eigvals_only,
                  verbose=verbose
            )

def cusolver_eigh_workspace_requirements(N, dtype):
    """
    Returns the workspace required bytes for cusolver cusolverDnXsyevdx for the 
    specified problem size and data type. Only torch.float32 and torch.float64 are
    supported.

    Parameters:
    N (torch::Tensor): The size of the symmetric matrix to extract eigenvalues/eigenvectors from
    dtype: The data type of the matrix
    
    Returns:
    workspaceInBytesDevice, workspaceInBytesHost
    """

    return syevdx_workspace_query(N, dtype)

#TODO: update these docs
def cusolver_batched_eigh(
        a,
        overwrite_a = False,
        lower = True, 
        eigvals_only = False,
        verbose = False
):
    """
    Uses cusolver cusolverDnXsyev_batched to get all the eigenvalues / eigenvectors
    of a batch of systems (batch_size, N, N). The base call
    to cusolverDnXsyev_batched overwrites the 'a' matrix and additionally allocates a workspace.
    If overwrite_a is False, then the a matrix is cloned and returned.

    Parameters:
    a (torch::Tensor): The matrix
    overwrite_a: Avoids a copy of the a tensor when true
    lower: Use the lower triangle, if false, use the upper triangle of the symmetric matrix
    eigvals_only: Only return the eigenvalues of a. If overwrite_a is True, then
        'a' will not be copied and this routine will still destroy 'a'.
    verbose: Currently only prints the requested workspace size. This is useful if you are crashing.

    Returns:
    eigenvalues, eigenvectors if eigvals_only is false
    eigenvalues of the range specified if eigvals_only is true
    """

    return syev_batched(a, 
                  overwrite_a=overwrite_a, 
                  lower=lower, 
                  eigvals_only=eigvals_only,
                  verbose=verbose
            )

def cusolver_batched_eigh_workspace_requirements(N, batch_size, dtype):
    """
    Returns the workspace required bytes for cusolver cusolverDnXsyev_batched for the 
    specified problem size and data type. Only torch.float32 and torch.float64 are
    supported.

    Parameters:
    N (torch::Tensor): The size of the symmetric matrix to extract eigenvalues/eigenvectors from
    batch_size (torch::Tensor): The batch size
    dtype: The data type of the matrix
    
    Returns:
    workspaceInBytesDevice, workspaceInBytesHost
    """

    return syev_batched_workspace_query(N, batch_size, dtype)

def cusolver_mg_eigh(
    a,
    overwrite_a = False,
    max_num_devices = 16,
    verbose = False
):
    """
    Uses cusolver cusolverMgsyevd to get all of the eigenvalues and eigenvectors of
    a system. It uses all nvidia devices visible to split the compute work and share 
    the working space. The base call of this function overwrites the eigenvectors in 
    to the 'a' matrix. When 'overwrite_a' is True, the vectors are returned in to 
    the passed in 'a' matrix

    Parameters:
    a (torch::Tensor): The matrix
    overwrite_a: Avoids a copy of the 'a' tensor when true
    
    Returns:
    eigenvalues, eigenvectors
    """
    return mgSyevd(a, overwrite_a=overwrite_a, max_num_devices=max_num_devices, verbose=verbose)

def cusolver_mg_eigh_workspace_requirements(N, dtype, num_devices=None, verbose=False):
    """
    Returns the workspace required bytes for cusolver cusolverDnXsyevdx for the 
    specified problem size and data type. Only torch.float32 and torch.float64 are
    supported.

    Parameters:
    N (torch::Tensor): The size of the symmetric matrix to extract eigenvalues/eigenvectors from
    dtype: The data type of the matrix
    num_devices: The amount of devices, if None, the number of visible devices to the system is used
    verbose: Print some stuff
    
    Returns:
    workspaceInBytesDevice
    """

    return mgSyevd_workspace_query(N, num_devices, dtype, verbose)
