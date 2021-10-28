import cookie from "cookie";

export async function redirectIfUnauthenticated(
  req: any
): Promise<{ destination: string; permanent: boolean } | null> {
  return new Promise(async (resolve) => {
    const REFRESH_TOKEN_NAME = "refresh-token";
    const referer = req.url;
    const redirect = {
      destination: `/login?referer=${referer}`,
      permanent: false,
    };

    const cookies = req.cookies;

    // Redundant undefinded check since req.cookies defaults to {}.
    // However, do need to check if REFRESH_TOKEN_NAME exists on the object.
    if (cookies === undefined || cookies[REFRESH_TOKEN_NAME] === undefined) {
      resolve(redirect);
    }

    const requestOptions = {
      method: "GET",
      headers: {
        Cookie: cookie.serialize(
          REFRESH_TOKEN_NAME,
          cookies[REFRESH_TOKEN_NAME]
        ),
      },
    };

    const res = await fetch(
      `http://${process.env.AUTH_SERVER_HOSTNAME}:${process.env.AUTH_SERVER_PORT}/userinfo`,
      requestOptions
    );

    const data = await res.json();

    if (!res.ok || data.success === false) {
      resolve(redirect);
    }

    resolve(null);
  });
}
