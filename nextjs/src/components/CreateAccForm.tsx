import React from "react";
import { useForm } from "../hooks/useForm";

interface Props {}

export const CreateAccForm: React.FC<Props> = () => {
  const initialState = {
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  } as {
    username: string;
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
    values: {
      username: String;
      email: String;
      password: String;
      confirmPassword: string;
    };
  } = useForm(createAccCallback, initialState);

  // a submit function that will execute upon form submission
  async function createAccCallback() {
    if (values.password !== values.confirmPassword) {
      console.error("Passwords do not match!");
    }

    console.log(
      `Username: ${values.username}\nEmail: ${values.email}\nPassword: ${values.password}`
    );

    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: values.username,
        email: values.email,
        password: values.password,
      }),
    };

    fetch(`/auth/add-user`, requestOptions)
      .then((res) => res.json())
      .then(
        (res) => {
          console.log(res);
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
          type="text"
          name="username"
          placeholder="Username"
          onChange={onChange}
        />
      </div>
      <div>
        <input
          type="email"
          name="email"
          placeholder="Email"
          onChange={onChange}
        />
      </div>
      <div>
        <input
          type="password"
          name="password"
          placeholder="Password"
          onChange={onChange}
        />
      </div>
      <div>
        <input
          type="password"
          name="confirmPassword"
          placeholder="Confirm password"
          onChange={onChange}
        />
      </div>
      <button type="submit">Submit</button>
    </form>
  );
};
