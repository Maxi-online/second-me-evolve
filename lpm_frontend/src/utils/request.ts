/* eslint-disable no-console */
import type { AxiosRequestConfig, AxiosRequestHeaders, AxiosResponse } from 'axios';
import axios from 'axios';

const service = axios.create({
  baseURL: '',
  timeout: 600 * 1000, // Timeout in milliseconds
  headers: {
    'Content-Type': 'application/json', //'application/json;charset=UTF-8',
    lang: 'en-US'.toLowerCase()
  },
  withCredentials: true // Allow cross-domain requests with cookies
});

export function Request<T = unknown, D = unknown>(configParam: AxiosRequestConfig) {
  return service.request<T, AxiosResponse<T>, D>(configParam);
}

export function POST<T = unknown>(configParam: AxiosRequestConfig) {
  return service.request<T, AxiosResponse<T>>({
    ...configParam,
    method: 'post'
  });
}

export function GET<T = unknown>(configParam: AxiosRequestConfig) {
  return service.request<T, AxiosResponse<T>>({
    ...configParam,
    method: 'get'
  });
}

export function DELETE<T = unknown>(configParam: AxiosRequestConfig) {
  return service.request<T, AxiosResponse<T>>({
    ...configParam,
    method: 'delete'
  });
}

/**
 * HTTP request interceptor
 */
service.interceptors.request.use(
  (config) => {
    const headers = config.headers ?? {};
    const timestamp = Date.now().toString();

    if (headers['Content-Type'] === 'application/json') {
      config.data = JSON.stringify(config.data);
    }

    if (!localStorage.getItem('upload') && window.location.pathname !== '/') {
      return Promise.reject('Please login first').finally(() => {
        setTimeout(() => {
          window.location.href = '/';
        }, 500);
      });
    }

    config.headers = {
      timestamp: timestamp,
      ...config.headers
    } as unknown as AxiosRequestHeaders;

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * HTTP response interceptor
 */
service.interceptors.response.use(
  (response) => {
    return checkStatus(response);
  },
  (error) => {
    console.log('Request error:', error);

    if (error.response) {
      console.log('Response error data:', error.response.data);
      console.log('Response error status:', error.response.status);
      console.log('Response error headers:', error.response.headers);
    } else if (error.request) {
      console.log('No response received:', error.request);
    } else {
      console.log('Error setting up request:', error.message);
    }

    throw error;
  }
);

const checkStatus = (response: any) => {
  // server check for custom code

  // http check
  if (response.status >= 200 && response.status < 300) {
    // HTTP normal
    return response;
  }

  return response.json();
};
