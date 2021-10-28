import React from "react";
import { useForm } from "../hooks/useForm";

interface Props {}

export const LoginForm: React.FC<Props> = () => {
  const initialState = {
    email: "",
    password: "",
  } as {
    email: string;
    password: string;
  };

  const {
    onChange,
    onSubmit,
    values,
  }: {
    onChange: any;
    onSubmit: any;
    values: { email: String; password: String };
  } = useForm(loginUserCallback, initialState);

  // a submit function that will execute upon form submission
  async function loginUserCallback() {
    console.log(`Email: ${values.email}\nPassword: ${values.password}`);

    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: values.email, password: values.password }),
    };

    fetch(`/auth/login`, requestOptions)
      .then((res) => res.json())
      .then(
        (res) => {
          console.log(res);
          if (res.success === true) {
            const urlParams = new URLSearchParams(window.location.search);
            const referer = urlParams.get("referer");
            const newLocation = referer === null ? "/" : referer;
            window.location.href = newLocation;
          } else {
          }
        },
        (error) => {
          console.log(error);
        }
      );
  }

  return (
    <form onSubmit={onSubmit}>
      <div>
        <input
          type="email"
          name="email"
          placeholder="email"
          onChange={onChange}
        />
      </div>
      <div>
        <input
          type="password"
          name="password"
          placeholder="password"
          onChange={onChange}
        />
      </div>
      <div>
        <input type="hidden" name="csrf-token" />
      </div>

      <button type="submit">Submit</button>
    </form>
  );
};
