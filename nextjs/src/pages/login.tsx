import React from "react";
import { LoginForm } from "../components/LoginForm";
import Link from "next/link";

export default function Login() {
  return (
    <>
      <LoginForm />
      <div>
        <Link href="/createAccount">
          <a>Create Account</a>
        </Link>
      </div>
      <div>
        <Link href="/forgotPassword">
          <a>Forgot Password?</a>
        </Link>
      </div>
    </>
  );
}
