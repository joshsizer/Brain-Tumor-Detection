import React from "react";
import { CreateAccForm } from "../components/CreateAccForm";
import Link from "next/link";

// **UNCOMMENT to enable account creation only when a user is
// logged in**
// import { GetServerSideProps } from "next";
// import { redirectIfUnauthenticated } from "../util/util";

// //https://nextjs.org/docs/basic-features/data-fetching#provided-req-middleware-in-getserversideprops
// export const getServerSideProps: GetServerSideProps = async ({ req }) => {
//   const redirect = await redirectIfUnauthenticated(req);
//   if (redirect !== null) {
//     return {
//       redirect,
//     };
//   }
//   return {
//     props: {}, // will be passed to the page component as props
//   };
// };

export default function CreateAccount() {
  return (
    <>
      <CreateAccForm />
      <a>Have an Account?</a>
      <Link href="/login"> Login.</Link>
    </>
  );
}
