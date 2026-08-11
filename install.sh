#!/usr/bin/env sh
set -eu

REPO_OWNER="cuzfrog"
REPO_NAME="docent-mcp"
BINARY_NAME="docent"
DEFAULT_INSTALL_DIR="${HOME}/.local/bin"

detect_target() {
    kernel=$(uname -s | tr '[:upper:]' '[:lower:]')
    machine=$(uname -m)
    case "${kernel}/${machine}" in
        linux/x86_64)
            printf 'x86_64-unknown-linux-gnu\n'
            ;;
        *)
            printf 'Unsupported platform: %s %s\n' "${kernel}" "${machine}" >&2
            exit 1
            ;;
    esac
}

build_download_url() {
    target=$1
    version=${DOCENT_VERSION:-latest}
    if [ "${version}" = "latest" ]; then
        printf 'https://github.com/%s/%s/releases/latest/download/%s-%s\n' \
            "${REPO_OWNER}" "${REPO_NAME}" "${BINARY_NAME}" "${target}"
    else
        printf 'https://github.com/%s/%s/releases/download/%s/%s-%s\n' \
            "${REPO_OWNER}" "${REPO_NAME}" "${version}" "${BINARY_NAME}" "${target}"
    fi
}

ensure_install_dir() {
    install_dir=$1
    if [ ! -d "${install_dir}" ]; then
        mkdir -p "${install_dir}"
    fi
}

add_install_dir_to_path_in_shell_profiles() {
    install_dir=$1
    if [ "${install_dir}" != "${DEFAULT_INSTALL_DIR}" ]; then
        return
    fi
    path_line='export PATH="$HOME/.local/bin:$PATH"'
    for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
        if [ -f "${rc}" ] && ! grep -qF "${path_line}" "${rc}"; then
            printf '\n# Added by docent installer\n%s\n' "${path_line}" >> "${rc}"
        fi
    done
}

main() {
    if [ -n "${RELEASE_URL:-}" ]; then
        download_url="${RELEASE_URL}"
    else
        target=$(detect_target)
        download_url=$(build_download_url "${target}")
    fi

    install_dir="${INSTALL_DIR:-${DEFAULT_INSTALL_DIR}}"
    ensure_install_dir "${install_dir}"

    temp_file=$(mktemp)
    trap 'rm -f "${temp_file}"' EXIT

    printf 'Downloading %s...\n' "${download_url}"
    curl -fsSL --retry 3 --retry-connrefused "${download_url}" -o "${temp_file}"

    chmod +x "${temp_file}"
    mv -f "${temp_file}" "${install_dir}/${BINARY_NAME}"

    add_install_dir_to_path_in_shell_profiles "${install_dir}"

    printf '%s installed to %s\n' "${BINARY_NAME}" "${install_dir}"
    printf 'Run "%s --version" to verify.\n' "${install_dir}/${BINARY_NAME}"

    if ! "${install_dir}/${BINARY_NAME}" --version; then
        printf 'Installation verification failed.\n' >&2
        exit 1
    fi

    case ":${PATH}:" in
        *":${install_dir}:"*) ;;
        *)
            printf '\nWarning: %s is not in your PATH.\n' "${install_dir}"
            printf 'Add the following line to your shell profile and restart your shell:\n'
            printf '  export PATH="%s:$PATH"\n' "${install_dir}"
            ;;
    esac
}

main "$@"
